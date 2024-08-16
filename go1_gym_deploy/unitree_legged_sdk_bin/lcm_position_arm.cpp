/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "unitree_legged_sdk/joystick.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <thread>
#include <lcm/lcm-cpp.hpp>
#include "state_estimator_lcmt.hpp"
#include "leg_control_data_lcmt.hpp"
#include "pd_tau_targets_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include <fstream> // 包含文件流头文件


#include "utility.h"
#include "Hardware/can.h"
#include "Hardware/motor.h"
#include "App/arm_control.h"
#include "App/keyboard.h"
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <atomic>
#include <cmath>
#include <iostream>

using namespace std;
using namespace UNITREE_LEGGED_SDK;




class Custom
{
public:
    Custom(uint8_t level){}

    void RobotControl();
    void init();
    void handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt * msg);
    void _simpleLCMThread();

    LowCmd cmd = {0};
    LowState state = {0};
    float qInit[3]={0};
    float qDes[3]={0};
    float sin_mid_q[3] = {0.0, 1.2, -2.0};
    float Kp[3] = {0};
    float Kd[3] = {0};
    double time_consume = 0;
    int rate_count = 0;
    int sin_count = 0;
    int motiontime = 0;
    float dt = 0.002;     // 0.001~0.01

    lcm::LCM _simpleLCM;
    std::thread _simple_LCM_thread;
    bool _firstCommandReceived;
    bool _firstRun;
    state_estimator_lcmt body_state = {0};
    leg_control_data_lcmt joint_state = {0};
    pd_tau_targets_lcmt joint_command = {0};
    rc_command_lcmt rc_command = {0};
    xRockerBtnDataStruct _keyData;
    int mode = 0;


    arx_arm ARX_ARM;
    can CAN_Handlej;
    float arm_joint_states[7] = {0}; // 接收手的状
    float arm_joint_command[7] = {0}; // 暂存 policy 输出的 actions

    std::ofstream outputFile;

    bool stop_flag;
    char key;

};

void Custom::init()
{
    _simpleLCM.subscribe("pd_plustau_targets", &Custom::handleActionLCM, this);
    _simple_LCM_thread = std::thread(&Custom::_simpleLCMThread, this);

    _firstCommandReceived = false;
    _firstRun = true;
    stop_flag = false;
    
    // init
    CAN_Handlej.Send_moto_Cmd1(1, 0, 12, 0, 0, 0);
    CAN_Handlej.Send_moto_Cmd1(2, 0, 12, 0, 0, 0);usleep(200);
    CAN_Handlej.Send_moto_Cmd1(4, 0, 12, 0, 0, 0);usleep(200);
    CAN_Handlej.Send_moto_Cmd2(5, 0, 1, 0, 0, 0);usleep(200);
    CAN_Handlej.Send_moto_Cmd2(6, 0, 1, 0, 0, 0);usleep(200);
    CAN_Handlej.Send_moto_Cmd2(7, 0, 1, 0, 0, 0);usleep(200);
    CAN_Handlej.Send_moto_Cmd2(8, 0, 1, 0, 0, 0);usleep(200);
    ARX_ARM.get_joint();
    ARX_ARM.target_pos_temp[0]=ARX_ARM.current_pos[0];
    ARX_ARM.target_pos_temp[1]=ARX_ARM.current_pos[1];
    ARX_ARM.target_pos_temp[2]=ARX_ARM.current_pos[2];
    ARX_ARM.target_pos_temp[3]=ARX_ARM.current_pos[3];
    ARX_ARM.target_pos_temp[4]=ARX_ARM.current_pos[4];
    ARX_ARM.target_pos_temp[5]=ARX_ARM.current_pos[5];
    ARX_ARM.target_pos_temp[6]=ARX_ARM.current_pos[6];

    // target
    joint_command.q_des[0] = 0;
    joint_command.q_des[1] = 0;
    joint_command.q_des[2] = 0;
    joint_command.q_des[3] = 0;
    joint_command.q_des[4] = 0;
    joint_command.q_des[5] = 0;
    joint_command.q_des[6] = 0;
    joint_command.q_des[7] = 0;

    for(int j = 0; j<7; j++){
        ARX_ARM.target_pos[j]=joint_command.q_des[j];
    }
    for(int i=0;i < 1000;i++)
    {
        ARX_ARM.get_joint();
        ARX_ARM.update_real();
		usleep(4200);
    }


    ARX_ARM.get_joint();
    
    joint_command.q_des[0]=ARX_ARM.current_pos[0];
    joint_command.q_des[1]=ARX_ARM.current_pos[1];
    joint_command.q_des[2]=ARX_ARM.current_pos[2];
    joint_command.q_des[3]=ARX_ARM.current_pos[3];
    joint_command.q_des[4]=ARX_ARM.current_pos[4];
    joint_command.q_des[5]=ARX_ARM.current_pos[5];
    joint_command.q_des[6]=ARX_ARM.current_pos[6];

    printf("reset motor\n");

    // outputFile.open("real.txt");
    // // 检查文件是否成功打开
    // if (!outputFile.is_open()) {
    //     std::cerr << "无法打开文件！" << std::endl;
    // }

}


double jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    p = initPos*(1-rate) + targetPos*rate;
    return p;
}

void Custom::handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt * msg){
    (void) rbuf;
    (void) chan;

    joint_command = *msg;
    _firstCommandReceived = true;

}

void Custom::_simpleLCMThread(){
    while(true){
        _simpleLCM.handle();
    }
}

void Custom::RobotControl()
{
    motiontime++;
    
    ARX_ARM.get_joint();

    // 传输机械臂的关节状态
    std::cout <<"-------------------------------------------------------"<< std::endl;
    
    for(int i = 0; i < 6; i++){
        joint_state.q_arm[i] = ARX_ARM.current_pos[i];
        std::cout << "机械臂位置状态：arm_joint_states[" << i <<"]: " << ARX_ARM.current_pos[i] << std::endl;

    }
    std::cout << std::endl;

    if (key == 's'){
        stop_flag = true;
        for (int i = 0; i<7; i++){
            ARX_ARM.target_pos[i] = ARX_ARM.current_pos[i];
            ARX_ARM.target_pos_temp[i] = ARX_ARM.current_pos[i];
        }
        printf("================ stop ==================");
    
    }
    else if (key == 'm'){
        printf("================ moving ==================");
        stop_flag = false;
        joint_state.q_arm[7] += 1.;
        if (joint_state.q_arm[7] >= 3){
            joint_state.q_arm[7] = 0;
        }
    }

    _simpleLCM.publish("state_estimator_data", &body_state);
    _simpleLCM.publish("leg_control_data", &joint_state);
    _simpleLCM.publish("rc_command", &rc_command);

    if (!stop_flag){

        for(int i = 0; i < 7; i++){  // 只是控制6个关节
                ARX_ARM.target_pos[i] =  joint_command.q_arm_des[i];
            // if(i == 1){
            // joint_command.q_des[i] /= 1.5;
            // ARX_ARM.target_pos[i] /= 1.5;
            // }
            // joint_command.q_des[i] = std::min(joint_command.q_des[i], 6.);
            // joint_command.q_des[i] = std::max(joint_command.q_des[i], -6.);
            std::cout << "机械臂控制指令：arm_joint_command[" << i <<"]: " << ARX_ARM.target_pos[i] << std::endl;

        }
        ARX_ARM.update_real();
    }
    else if (stop_flag) {
        
        ARX_ARM.target_pos_temp[0] = ARX_ARM.ramp(ARX_ARM.target_pos[0], ARX_ARM.target_pos_temp[0],0.001);
        ARX_ARM.target_pos_temp[1] = ARX_ARM.ramp(ARX_ARM.target_pos[1], ARX_ARM.target_pos_temp[1],0.001);
        ARX_ARM.target_pos_temp[2] = ARX_ARM.ramp(ARX_ARM.target_pos[2], ARX_ARM.target_pos_temp[2],0.001);
        ARX_ARM.target_pos_temp[3] = ARX_ARM.ramp(ARX_ARM.target_pos[3], ARX_ARM.target_pos_temp[3],0.001);
        ARX_ARM.target_pos_temp[4] = ARX_ARM.ramp(ARX_ARM.target_pos[4], ARX_ARM.target_pos_temp[4],0.001);
        ARX_ARM.target_pos_temp[5] = ARX_ARM.ramp(ARX_ARM.target_pos[5], ARX_ARM.target_pos_temp[5],0.001);
        ARX_ARM.target_pos_temp[6] = ARX_ARM.ramp(ARX_ARM.target_pos[6], ARX_ARM.target_pos_temp[6],0.001);

        CAN_Handlej.Send_moto_Cmd1(1, 150, 12, ARX_ARM.target_pos_temp[0], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd1(2, 150, 12, ARX_ARM.target_pos_temp[1], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd1(4, 150, 12, ARX_ARM.target_pos_temp[2], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(5, 30, 0.8, ARX_ARM.target_pos_temp[3], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(6, 25, 0.8, ARX_ARM.target_pos_temp[4], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(7, 10, 1,   ARX_ARM.target_pos_temp[5], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(8, 10, 1,   ARX_ARM.target_pos_temp[6], 0, 0);usleep(200);
    }

    // if (joint_command.q_des[6] == 0){
    //     ARX_ARM.update_real();
    //     printf("Pose Control.");
    // }
    // else if(joint_command.q_des[6] == 1){
    //     CAN_Handlej.Send_moto_Cmd1(1, 0, 12, 0, 0, joint_command.q_des[0]);
    //     CAN_Handlej.Send_moto_Cmd1(2, 0, 12, 0, 0, joint_command.q_des[1]);usleep(200);
    //     CAN_Handlej.Send_moto_Cmd1(4, 0, 12, 0, 0, joint_command.q_des[2]);usleep(200);
    //     CAN_Handlej.Send_moto_Cmd2(5, 0, 1, 0, 0,  joint_command.q_des[3]);usleep(200);
    //     CAN_Handlej.Send_moto_Cmd2(6, 0, 1, 0, 0,  joint_command.q_des[4]);usleep(200);
    //     CAN_Handlej.Send_moto_Cmd2(7, 0, 1, 0, 0,  joint_command.q_des[5]);usleep(200);
    //     printf("Torque Control.");
    // }
    // else{
    //     printf("No defined Control.");
    //     return;
    // }

    usleep(4200);
   // 写入文本到文件
    // if (outputFile.is_open()){
    //     for(int i = 0 ; i < 6; i++){
    //         outputFile << ARX_ARM.target_pos_temp[i] << " ";
    //     }
    // }
    // outputFile << std::endl;

}

int main(void)
{
    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    arx5_keyboard ARX_KEYBOARD;

    Custom custom(LOWLEVEL);
    custom.init();
    std::thread keyThread(&arx5_keyboard::detectKeyPress, &ARX_KEYBOARD);

    // InitEnvironment();
    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));

    loop_control.start();


    while(1){
        char key = ARX_KEYBOARD.keyPress.load();
        custom.key = key;
        if(key=='q'){
            break;
        }
		usleep(4200);

    };



    // custom.ARX_ARM.target_pos[0]=custom.ARX_ARM.current_pos[0];
    // custom.ARX_ARM.target_pos[1]=custom.ARX_ARM.current_pos[1];
    // custom.ARX_ARM.target_pos[2]=custom.ARX_ARM.current_pos[2];
    // custom.ARX_ARM.target_pos[3]=custom.ARX_ARM.current_pos[3];
    // custom.ARX_ARM.target_pos[4]=custom.ARX_ARM.current_pos[4];
    // custom.ARX_ARM.target_pos[5]=custom.ARX_ARM.current_pos[5];
    // custom.ARX_ARM.target_pos[6]=custom.ARX_ARM.current_pos[6];

    // custom.ARX_ARM.target_pos_temp[0]=custom.ARX_ARM.current_pos[0];
    // custom.ARX_ARM.target_pos_temp[1]=custom.ARX_ARM.current_pos[1];
    // custom.ARX_ARM.target_pos_temp[2]=custom.ARX_ARM.current_pos[2];
    // custom.ARX_ARM.target_pos_temp[3]=custom.ARX_ARM.current_pos[3];
    // custom.ARX_ARM.target_pos_temp[4]=custom.ARX_ARM.current_pos[4];
    // custom.ARX_ARM.target_pos_temp[5]=custom.ARX_ARM.current_pos[5];
    // custom.ARX_ARM.target_pos_temp[6]=custom.ARX_ARM.current_pos[6];

    loop_control.shutdown();


    for(int i=0; i<1000; i++){
        custom.ARX_ARM.get_joint();
        // custom.ARX_ARM.update_real();
        std::cout << "damping" << std::endl;
        custom.CAN_Handlej.Send_moto_Cmd1(1, 0, 12, 0, 0, 0);
        custom.CAN_Handlej.Send_moto_Cmd1(2, 0, 12, 0, 0, 0);usleep(200);
        custom.CAN_Handlej.Send_moto_Cmd1(4, 0, 12, 0, 0, 0);usleep(200);
        custom.CAN_Handlej.Send_moto_Cmd2(5, 0, 1, 0, 0, 0);usleep(200);
        custom.CAN_Handlej.Send_moto_Cmd2(6, 0, 1, 0, 0, 0);usleep(200);
        custom.CAN_Handlej.Send_moto_Cmd2(7, 0, 1, 0, 0, 0);usleep(200);
        custom.CAN_Handlej.Send_moto_Cmd2(8, 0, 1, 0, 0, 0);usleep(200);

		usleep(4200);
    }
    printf("######################## Finish reset! ########################");

    custom.CAN_Handlej.Send_moto_Cmd1(1, 0, 12, 0, 0, 0);
    custom.CAN_Handlej.Send_moto_Cmd1(2, 0, 12, 0, 0, 0);usleep(200);
    custom.CAN_Handlej.Send_moto_Cmd1(4, 0, 12, 0, 0, 0);usleep(200);
    custom.CAN_Handlej.Send_moto_Cmd2(5, 0, 1, 0, 0, 0);usleep(200);
    custom.CAN_Handlej.Send_moto_Cmd2(6, 0, 1, 0, 0, 0);usleep(200);
    custom.CAN_Handlej.Send_moto_Cmd2(7, 0, 1, 0, 0, 0);usleep(200);
    custom.CAN_Handlej.Send_moto_Cmd2(8, 0, 1, 0, 0, 0);usleep(200);
    usleep(4200);

    custom.outputFile.close();
    return 0;
}