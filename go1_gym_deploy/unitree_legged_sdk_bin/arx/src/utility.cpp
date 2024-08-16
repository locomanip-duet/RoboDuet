#include "utility.h"

float valid_angle(float a)
{
    return (a - (floor((a + M_PI) / M_2PI) * M_2PI));
}

float angle_diff(float a, float b)
{
    float dif = a - b;
    if (dif > M_PI)
    {
        return dif - M_2PI;
    }
    else if (dif < -M_PI)
    {
        return dif + M_2PI;
    }
    else
    {
        return dif;
    }
}

pid::pid()
{
}

pid::~pid()
{
}

void pid::init(float k[3], float integral_max, float out_max)
{
    Kp = k[0];
    Ki = k[1];
    Kd = k[2];
    vout = 0.0;
    outMax = out_max;
    integralMax = integral_max;
}

float pid::calc(float target, float current)
{
    float error = target - current;
    integral_error = limit(integral_error + error, -integralMax, integralMax);
    float out = Kp * error + Ki * integral_error + Kd * (error - last_error);
    last_error = error;
    vout = limit(out, -outMax, outMax);
    return vout;
}

void pid::clear(void)
{
    integral_error = 0.0;
    last_error = 0.0;
    vout = 0.0;
}

LowPassFilter::LowPassFilter(float sample_freq_, float cut_freq_){
    float ohm = tanf(M_PI * cut_freq_ / sample_freq_);
    float c = 1.0f + 2.0f * cosf(M_PI / 4.0f) * ohm + ohm * ohm;
    a[0] = ohm * ohm / c;
    a[1] = 2.0f * a[0];
    a[2] = a[0];
    b[0] = 2.0f * (ohm * ohm - 1.0f) / c;
    b[1] = (1.0f - 2.0f * cosf(M_PI / 4.0f) * ohm + ohm * ohm) / c;
}

float LowPassFilter::clac(float new_data_){
    out[0] = a[0] * new_data_ + a[1] * input[0] + a[2] * input[1] - b[0] * out[1] - b[1] * out[2];
    input[1] = input[0];
    input[0] = new_data_;
    out[2] = out[1];
    out[1] = out[0];
    return out[0];
}