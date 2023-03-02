#include "PIDAimer.h"
#include <iostream>
#include <Windows.h>
#include "defines.h"


using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

PID::PID() {};
PID::PID(float kp, float ki, float kd) : kp(kp), ki(ki), kd(kd), prev_err(0), integral(0), last_step(0) {};

float PID::step(int err) {
	float now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000;
	float dt = now - last_step;
	if (dt > 0.1) dt = 0.1;
	last_step = now;
	float derivative = (float)err - prev_err;
	if (dt > 0.0001) {
		integral += err * dt;
		derivative /= dt;
	}
	prev_err = err;
	return kp * err + ki * integral + kd * derivative;
}

void PID::clear() {
	prev_err = integral = last_step = 0;
}

// PIDAimer::PIDAimer() : PIDAimer::PIDAimer(0.4, 0.02, 0.01) {};
PIDAimer::PIDAimer() : PIDAimer::PIDAimer(0.4, 0.15, 0.11) {};

PIDAimer::PIDAimer(float kp, float ki, float kd) {
	std::cout << "kp: " << kp << ", ki: " << ki << ", kd: " << kd << std::endl;
	pidx = PID(kp, ki, kd);
	pidy = PID(kp, ki, kd);
	maxX = maxY = 30;
	y_last_sway_time = std::chrono::system_clock::now();
	x_last_sway_time = std::chrono::system_clock::now();

};

void PIDAimer::aim(int dx, int dy, int width, int height) {
	
	dx += (int)(X_SWAY_RANGE * width);
	dy += (int)(Y_SWAY_RANGE * height);
	dx -= CENTER;
	dy -= CENTER;
	if (-width*0.5 <= dx && dx <= width*0.5 && -height*0.5 <= dy && dy <= height*0.5) { return; }
	float mx = pidx.step(dx);
	float my = pidy.step(dy);
	if (mx > maxX) mx = maxX;
	if (mx < -maxX) mx = -maxX;
	if (my > maxY) my = maxY;
	if (my < -maxY) my = -maxY;
	auto now = std::chrono::system_clock::now();
	auto y_duration = duration_cast<milliseconds>(now - y_last_sway_time);
	if (y_duration.count()/300 > Y_SWAY_INTERVAL) {
		y_last_sway_time = now;
		Y_SWAY_INTERVAL = (float)(rand() % 16 + 10) / 100.0;
		Y_SWAY_RANGE = ((float)rand() / RAND_MAX) * 0.7 - 0.3;
	}
	auto x_duration = duration_cast<milliseconds>(now - x_last_sway_time);
	if (x_duration.count()/450 > X_SWAY_INTERVAL) {
		x_last_sway_time = now;
		X_SWAY_INTERVAL = (float)(rand() % 151 + 50) / 1000.0;
		X_SWAY_RANGE = ((float)rand() / RAND_MAX) * 0.6f - 0.3f;
	}
	
	mouse_event(MOUSEEVENTF_MOVE, (int)mx, (int)my, 0, 0);
}
