#pragma once

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/directx.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "PIDAimer.h"
#include "detector.h"
#include <chrono>
#include <atlstr.h>
#include <format>

class SimpleCapture
{
public:
    SimpleCapture(
        winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device,
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item,
        winrt::Windows::Graphics::DirectX::DirectXPixelFormat pixelFormat);
    ~SimpleCapture() { Close(); }

    void StartCapture();
    winrt::Windows::UI::Composition::ICompositionSurface CreateSurface(
        winrt::Windows::UI::Composition::Compositor const& compositor);

    bool IsCursorEnabled() { CheckClosed(); return m_session.IsCursorCaptureEnabled(); }
	void IsCursorEnabled(bool value) { CheckClosed(); m_session.IsCursorCaptureEnabled(value); }
    bool IsBorderRequired() { CheckClosed(); return m_session.IsBorderRequired(); }
    void IsBorderRequired(bool value) { CheckClosed(); m_session.IsBorderRequired(value); }
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem CaptureItem() { return m_item; }
    void SetStatusBarHwnd(HWND statusHwnd) { statusBarHwnd = statusHwnd; }
    void SetPixelFormat(winrt::Windows::Graphics::DirectX::DirectXPixelFormat pixelFormat)
    {
        CheckClosed();
        auto newFormat = std::optional(pixelFormat);
        m_pixelFormatUpdate.exchange(newFormat);
    }

    void Close();

private:
    void OnFrameArrived(
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender,
        winrt::Windows::Foundation::IInspectable const& args);

    inline void CheckClosed()
    {
        if (m_closed.load() == true)
        {
            throw winrt::hresult_error(RO_E_CLOSED);
        }
    }

    void ResizeSwapChain();
    bool TryResizeSwapChain(winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame const& frame);
    bool TryUpdatePixelFormat();

private:
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem m_item{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool m_framePool{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession m_session{ nullptr };
    winrt::Windows::Graphics::SizeInt32 m_lastSize;

    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_device{ nullptr };

    winrt::com_ptr<IDXGISwapChain1> m_swapChain{ nullptr };
    winrt::com_ptr<ID3D11DeviceContext> m_d3dContext{ nullptr };
    winrt::Windows::Graphics::DirectX::DirectXPixelFormat m_pixelFormat;

    std::atomic<std::optional<winrt::Windows::Graphics::DirectX::DirectXPixelFormat>> m_pixelFormatUpdate = std::nullopt;

    std::atomic<bool> m_closed = false;
    std::atomic<bool> m_captureNextImage = false;
    winrt::com_ptr <ID3D11Device> m_d3dDevice{ nullptr };
    int frame_count = 0;
    D3D11_TEXTURE2D_DESC desc;
    ID3D11Texture2D* stagingTexture;
    ID3D11Texture2D* displayTexture;
    cv::ocl::Context m_oclCtx;
    cv::String m_oclPlatformName;
    cv::String m_oclDevName;
    YOLODetector detector{ nullptr };
    std::vector<Detection> result;
    std::vector<std::string> classNames;
    PIDAimer aimer = PIDAimer();
    Target target;
    float fps = -1;
    HWND statusBarHwnd;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
};