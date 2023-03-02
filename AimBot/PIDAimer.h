#pragma once
#include <chrono>
class PID {
public:
	PID();
	PID(float kp, float ki, float kd);
	float step(int err);
	void clear();
private:
	float kp = 0, ki = 0, kd = 0, prev_err = 0, integral = 0, last_step = 0;
};

class PIDAimer {
public:
	PIDAimer();
	PIDAimer(float kp, float ki, float kd);
	void aim(int dx, int dy, int width, int height);
private:
	PID pidx, pidy;
	int maxX, maxY;
	std::chrono::system_clock::time_point y_last_sway_time;
	std::chrono::system_clock::time_point x_last_sway_time;
	float Y_SWAY_INTERVAL = 0.2f;
	float X_SWAY_INTERVAL = 0.1f;
	float Y_SWAY_RANGE = 0;
	float X_SWAY_RANGE = 0;
};