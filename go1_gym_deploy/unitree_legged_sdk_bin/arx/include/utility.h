#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <cmath>
#include <vector>
#define M_2PI 6.28318530717958623200

template <typename T>
T limit(const T &v, const T &lb, const T &ub)
{
    if (v < lb)
    {
        return lb;
    }
    else if (v > ub)
    {
        return ub;
    }
    else
    {
        return v;
    }
}

template <typename T>
bool in_range(const T &v, const T &lb, const T &ub)
{
    if (v < lb)
    {
        return false;
    }
    else if (v > ub)
    {
        return false;
    }
    else
    {
        return true;
    }
}

float valid_angle(float a);

float angle_diff(float a, float b);

class pid
{
public:
    pid();
    ~pid();
    void init(float k[3], float integral_max, float out_max);
    float calc(float target, float current);
    void clear(void);
    float vout;

private:
    float Kp;
    float Ki;
    float Kd;
    float last_error;
    float integral_error;
    float integralMax;
    float outMax;
};

class LowPassFilter{
public:
    LowPassFilter(float sample_freq_, float cut_freq_);
    ~LowPassFilter(){};
    float clac(float new_data_);

private:
    float a[3], b[2];
    std::vector<float> input = std::vector<float>(2), out = std::vector<float>(3);
};

#endif
