#pragma once

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
	void aim(int dx, int dy);
private:
	PID pidx, pidy;
	int maxX, maxY;
};