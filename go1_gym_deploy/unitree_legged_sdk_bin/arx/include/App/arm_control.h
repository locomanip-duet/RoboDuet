#ifndef _ARM_CONTROL_H_
#define _ARM_CONTROL_H_

#include "utility.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../Hardware/can.h"
#include "../Hardware/motor.h"
#include <string.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <memory.h>
#include <dirent.h>
#include <vector>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <tinyxml2.h>
#include <string>

#define filter_torque 0.3f

class FIFO_Queue
{
public:
    FIFO_Queue(uint queue_size);
    ~FIFO_Queue() = default;
    uint count = 0;
    uint write_ptr = 0, read_ptr = 0;
};

class arx_arm
{
public:
    arx_arm();
    ~arx_arm()=default;

    float current_pos[7] = {};
    float current_vel[7] = {0.0};
    float current_torque[7] = {0.0f};
    float target_pos[7] = {0.0f}, last_target_pos[7] = {0.0f};
    float target_pos_temp[7] = {0.0f};
    float target_vel[7] = {0.0f};
    float ros_control_pos[7] ={};
    float ros_control_pos_t[7] ={};
    float ros_control_vel[7] ={};
    float ros_control_cur[7] ={};

    float lower_bound_waist[3] = {  0.0, -0.4, -0.4};
    float upper_bound_waist[3] = {  0.4,  0.4,  0.4};

    float lower_bound_pitch = -1.35;
    float upper_bound_pitch = M_PI/2; 
    float lower_bound_yaw = -1.35;
    float upper_bound_yaw = 1.35;
    float lower_bound_roll = -1.35;
    float upper_bound_roll = 1.35;


    float max_torque = 15;

    float ramp(float goal, float current, float ramp_k);

    void get_joint();

    void init_step();
    bool is_starting = true,is_arrived = false;

    float prev_target_pos[6] = {0};

    void calc_joint_acc();

    bool is_recording=false;
    std::string out_teach_path;

    float Lower_Joint[7] = { -3.14  ,0      ,-0.1   ,-1.671 ,-1.671,-1.57 ,0};
    float Upper_Joint[7] = { 2.618 , 3.14   ,3.24, 1.671 , 1.671, 1.57 ,4.2};

    void safe_model(void);
    void set_zero(void);
    void update(void);
    void update_real();
    void motor_control();
    void joint_control();
    void gripper_control(float gripper_spd);

    can CAN_Handlej;
    bool current_normal=true;
    int temp_current_normal=0;
    bool temp_condition=true;
    int temp_init=0;

    float gripper_cout=0,gripper_spd,gripper_max_cur=50;
    float ros_move_k_x=100.0f,ros_move_k_y=100.0f,ros_move_k_z=100.0f,ros_move_k_yaw=100.0f,ros_move_k_pitch=100.0f,ros_move_k_roll=100.0f;
    void arm_replay_mode();
    void arm_torque_mode();
    void arm_reset_mode();
    void arm_get_pos();
    void arm_teach_mode();  
    void limit_pos();   
    void cmd_init();
    int play_gripper=0;
    void limit_joint(float* Set_Pos);

private:

    float test_pos = 0;
    uint test_cnt = 0;
};

#endif