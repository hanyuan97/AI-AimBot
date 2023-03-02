#include "pch.h"
#include "SimpleCapture.h"
#include "defines.h"
#define KEY_DOWN(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x8000) ? 1:0)
#define KEYDOWN(vk_code) ((GetAsyncKeyState(vk_code) & 0x8000) ? 1 : 0)

namespace winrt
{
    using namespace Windows::Foundation;
    using namespace Windows::Foundation::Numerics;
    using namespace Windows::Graphics;
    using namespace Windows::Graphics::Capture;
    using namespace Windows::Graphics::DirectX;
    using namespace Windows::Graphics::DirectX::Direct3D11;
    using namespace Windows::System;
    using namespace Windows::UI;
    using namespace Windows::UI::Composition;
}

namespace util
{
    using namespace robmikh::common::uwp;
}

SimpleCapture::SimpleCapture(winrt::IDirect3DDevice const& device, winrt::GraphicsCaptureItem const& item, winrt::DirectXPixelFormat pixelFormat)
{
    m_item = item;
    m_device = device;
    m_pixelFormat = pixelFormat;

    auto d3dDevice = GetDXGIInterfaceFromObject<ID3D11Device>(m_device);
    d3dDevice->GetImmediateContext(m_d3dContext.put());

    m_swapChain = util::CreateDXGISwapChain(d3dDevice, static_cast<uint32_t>(m_item.Size().Width), static_cast<uint32_t>(m_item.Size().Height),
        static_cast<DXGI_FORMAT>(m_pixelFormat), 2);

    // Creating our frame pool with 'Create' instead of 'CreateFreeThreaded'
    // means that the frame pool's FrameArrived event is called on the thread
    // the frame pool was created on. This also means that the creating thread
    // must have a DispatcherQueue. If you use this method, it's best not to do
    // it on the UI thread. 
    m_framePool = winrt::Direct3D11CaptureFramePool::Create(m_device, m_pixelFormat, 2, m_item.Size());
    m_session = m_framePool.CreateCaptureSession(m_item);
    // m_session.IsBorderRequired(false);
    m_lastSize = m_item.Size();
    m_framePool.FrameArrived({ this, &SimpleCapture::OnFrameArrived });
    m_d3dDevice = d3dDevice;

    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc = { 1,0 };
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    if (cv::ocl::haveOpenCL())
    {
        m_oclCtx = cv::directx::ocl::initializeContextFromD3D11Device(m_d3dDevice.get());
    }
    m_oclDevName = cv::ocl::useOpenCL() ?
        cv::ocl::Context::getDefault().device(0).name() :
        "No OpenCL device";

    const bool isGPU = true;
    const std::string modelPath = "apex_0220_half.trt";
    TRTdetector = YOLOv5TRTDetector(modelPath);
    //const std::string modelPath = "apex_yolov8.trt";
    //V8detector = new YOLOv8(modelPath);
    //const std::string modelPath = "apex_yolov8_end2end_20_half.trt";
    //V8detector = new YOLOv8end2end(modelPath);

    //V8detector->make_pipe(true);

    classNames.push_back("per");
}

void SimpleCapture::StartCapture()
{
    CheckClosed();
    m_session.StartCapture();
    start = std::chrono::high_resolution_clock::now();
}

winrt::ICompositionSurface SimpleCapture::CreateSurface(winrt::Compositor const& compositor)
{
    CheckClosed();
    return util::CreateCompositionSurfaceForSwapChain(compositor, m_swapChain.get());
}

void SimpleCapture::Close()
{
    auto expected = false;
    if (m_closed.compare_exchange_strong(expected, true))
    {
        m_session.Close();
        m_framePool.Close();

        m_swapChain = nullptr;
        m_framePool = nullptr;
        m_session = nullptr;
        m_item = nullptr;
    }
}

void SimpleCapture::ResizeSwapChain()
{
    winrt::check_hresult(m_swapChain->ResizeBuffers(2, static_cast<uint32_t>(m_lastSize.Width), static_cast<uint32_t>(m_lastSize.Height),
        static_cast<DXGI_FORMAT>(m_pixelFormat), 0));
}

bool SimpleCapture::TryResizeSwapChain(winrt::Direct3D11CaptureFrame const& frame)
{
    auto const contentSize = frame.ContentSize();
    if ((contentSize.Width != m_lastSize.Width) ||
        (contentSize.Height != m_lastSize.Height))
    {
        // The thing we have been capturing has changed size, resize the swap chain to match.
        m_lastSize = contentSize;
        ResizeSwapChain();
        return true;
    }
    return false;
}

bool SimpleCapture::TryUpdatePixelFormat()
{
    auto newFormat = m_pixelFormatUpdate.exchange(std::nullopt);
    if (newFormat.has_value())
    {
        auto pixelFormat = newFormat.value();
        if (pixelFormat != m_pixelFormat)
        {
            m_pixelFormat = pixelFormat;
            ResizeSwapChain();
            return true;
        }
    }
    return false;
}

void SimpleCapture::OnFrameArrived(winrt::Direct3D11CaptureFramePool const& sender, winrt::IInspectable const&)
{
    auto swapChainResizedToFrame = false;
    {
        auto frame = sender.TryGetNextFrame();
        auto left = frame.ContentSize().Width / 2 - DETECTION_RANGE / 2;
        auto top = frame.ContentSize().Height / 2 - DETECTION_RANGE / 2;
        auto x2 = frame.ContentSize().Width / 2 + DETECTION_RANGE / 2;
        auto y2 = frame.ContentSize().Height / 2 + DETECTION_RANGE / 2;
        auto w = x2 - left + 1;
        auto h = y2 - top + 1;
        swapChainResizedToFrame = TryResizeSwapChain(frame);

        frame_count++;

        if (swapChainResizedToFrame) {
            // frame_count = 0;
            desc.Width = m_lastSize.Width;
            desc.Height = m_lastSize.Height;
            m_framePool.Recreate(m_device, m_pixelFormat, 2, m_lastSize);
        }
        
        desc.Width = frame.ContentSize().Width;
        desc.Height = frame.ContentSize().Height;
        // desc.Width = w;
        // desc.Height = h;

        m_d3dDevice.get()->CreateTexture2D(&desc, nullptr, &stagingTexture);
        auto surfaceTexture = GetDXGIInterfaceFromObject<ID3D11Texture2D>(frame.Surface());
        /*
        D3D11_TEXTURE2D_DESC desc2;
        surfaceTexture.get()->GetDesc(&desc2);
        
        DXGI_FORMAT format = desc2.Format;
        */
        m_d3dContext->CopyResource(stagingTexture, surfaceTexture.get());
        
        D3D11_MAPPED_SUBRESOURCE mappedTex;
        m_d3dContext->Map(stagingTexture, 0, D3D11_MAP_READ, 0, &mappedTex);
        m_d3dContext->Unmap(stagingTexture, 0);
        //cv::cuda::GpuMat frame_gpu = cv::cuda::GpuMat(desc.Height, desc.Width, CV_8UC4, mappedTex.pData, mappedTex.RowPitch);
        cv::Mat frame_cpu = cv::Mat(desc.Height, desc.Width, CV_8UC4, mappedTex.pData, mappedTex.RowPitch);
        cv::rectangle(frame_cpu, cv::Rect(left, top, w, h), cv::Scalar(0, 255, 0), 2);
        frame_cpu = frame_cpu(cv::Rect(left, top, w, h));
        desc.Width = w;
        desc.Height = h;
        
        winrt::com_ptr<ID3D11Texture2D> backBuffer;
        winrt::check_hresult(m_swapChain->GetBuffer(0, winrt::guid_of<ID3D11Texture2D>(), backBuffer.put_void()));
        m_d3dContext->CopyResource(backBuffer.get(), stagingTexture);
        
        
        auto detect_frame_start = std::chrono::high_resolution_clock::now();
        result = TRTdetector.detect(frame_cpu);
        /*V8detector->copy_from_Mat(frame_cpu);
        V8detector->infer();
        V8detector->postprocess(result);*/
        auto detect_frame_end = std::chrono::high_resolution_clock::now();
        
        

        detect_frame_time += std::chrono::duration_cast<std::chrono::milliseconds>(detect_frame_end - detect_frame_start).count();
        
        utils::findClosest(result, TARGET_CLASSID, target);
        if (target.isFind && (KEY_DOWN(VK_LBUTTON) || KEY_DOWN(VK_RBUTTON))) aimer.aim(target.pos.x, target.pos.y, target.box.width, target.box.height);
        auto draw_frame_start = std::chrono::high_resolution_clock::now();
        utils::visualizeDetection(frame_cpu, result, classNames, target);
        auto draw_frame_end = std::chrono::high_resolution_clock::now();
        draw_frame_time += std::chrono::duration_cast<std::chrono::milliseconds>(draw_frame_end - draw_frame_start).count();
        if (frame_count >= 100)
        {
            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();
            // std::cout << fps_label_str.c_str() << std::endl;
            start = std::chrono::high_resolution_clock::now();
            CString tmp = _T("test");
            tmp.Format(_T("FPS: %.2f, avg. detect time: %.2f, avg. draw frame time: %.2f"), fps, detect_frame_time/100, draw_frame_time/100);
            SendMessage(statusBarHwnd, WM_SETTEXT, 0, (LPARAM)(LPCTSTR)tmp);
            frame_count = 0;
            detect_frame_time = 0;
            draw_frame_time = 0;
        }
        
        stagingTexture->Release();
        

        DXGI_PRESENT_PARAMETERS presentParameters{};
        m_swapChain->Present1(1, 0, &presentParameters);

        swapChainResizedToFrame = swapChainResizedToFrame || TryUpdatePixelFormat();
    }
    if (swapChainResizedToFrame)
    {
        m_framePool.Recreate(m_device, m_pixelFormat, 2, m_lastSize);
    }

}