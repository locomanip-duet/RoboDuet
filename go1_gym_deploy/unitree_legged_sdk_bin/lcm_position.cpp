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

#include "utility.h"
#include "Hardware/can.h"
#include "Hardware/motor.h"
#include "App/arm_control.h"
#include "App/keyboard.h"

#include <cmath>
#include <iostream>

using namespace std;
using namespace UNITREE_LEGGED_SDK;

class Custom
{
public:
    Custom(uint8_t level): safe(LeggedType::Go1), udp(level, 8090, "192.168.123.10", 8007) {
        udp.InitCmdData(cmd);
    }
    void UDPRecv();
    void UDPSend();
    void RobotControl();
    void init();
    void handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt * msg);
    void _simpleLCMThread();

    Safety safe;
    UDP udp;
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
    state_estimator_lcmt body_state_simple = {0};
    leg_control_data_lcmt joint_state = {0};
    pd_tau_targets_lcmt joint_command = {0};
    rc_command_lcmt rc_command = {0};

    xRockerBtnDataStruct _keyData;
    int mode = 0;

    arx_arm ARX_ARM;
    can CAN_Handlej;
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

    // set nominal pose

    for(int i = 0; i < 12; i++){
        joint_command.qd_des[i] = 0;
        joint_command.tau_ff[i] = 0;
        joint_command.kp[i] = 35.;
        joint_command.kd[i] = 1.;
    }

    joint_command.q_des[0] = -0.3;
    joint_command.q_des[1] = 1.2;
    joint_command.q_des[2] = -2.721;
    joint_command.q_des[3] = 0.3;
    joint_command.q_des[4] = 1.2;
    joint_command.q_des[5] = -2.721;
    joint_command.q_des[6] = -0.3;
    joint_command.q_des[7] = 1.2;
    joint_command.q_des[8] = -2.721;
    joint_command.q_des[9] = 0.3;
    joint_command.q_des[10] = 1.2;
    joint_command.q_des[11] = -2.721;

    printf("SET DOG NOMINAL POSE");

    // // init arm
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

    joint_command.q_arm_des[0] = 0;
    joint_command.q_arm_des[1] = 0;
    joint_command.q_arm_des[2] = 0;
    joint_command.q_arm_des[3] = 0;
    joint_command.q_arm_des[4] = 0;
    joint_command.q_arm_des[5] = 0;
    joint_command.q_arm_des[6] = 0;
    
    for(int j = 0; j<7; j++){
        ARX_ARM.target_pos[j]=joint_command.q_arm_des[j];
    }
    for(int i=0;i < 1000;i++)
    {
        ARX_ARM.get_joint();
        ARX_ARM.update_real();
		usleep(4200);
    }


    ARX_ARM.get_joint();
    
    joint_command.q_arm_des[0]=ARX_ARM.current_pos[0];
    joint_command.q_arm_des[1]=ARX_ARM.current_pos[1];
    joint_command.q_arm_des[2]=ARX_ARM.current_pos[2];
    joint_command.q_arm_des[3]=ARX_ARM.current_pos[3];
    joint_command.q_arm_des[4]=ARX_ARM.current_pos[4];
    joint_command.q_arm_des[5]=ARX_ARM.current_pos[5];
    joint_command.q_arm_des[6]=ARX_ARM.current_pos[6];

    printf("SET ARM NOMINAL POSE");

}

void Custom::UDPRecv()
{
    udp.Recv();
}

void Custom::UDPSend()
{
    udp.Send();
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
    // std::cout<< joint_command.q_des[0] << std::endl;
}


void Custom::_simpleLCMThread(){
    while(true){
        _simpleLCM.handle();
    }
}

void Custom::RobotControl()
{
    motiontime++;
    udp.GetRecv(state);

    memcpy(&_keyData, &state.wirelessRemote[0], 40);
    ARX_ARM.get_joint();

    rc_command.left_stick[0] = _keyData.lx;
    rc_command.left_stick[1] = _keyData.ly;
    rc_command.right_stick[0] = _keyData.rx;
    rc_command.right_stick[1] = _keyData.ry;
    rc_command.right_lower_right_switch = _keyData.btn.components.R2;
    rc_command.right_upper_switch = _keyData.btn.components.R1;
    rc_command.left_lower_left_switch = _keyData.btn.components.L2;
    rc_command.left_upper_switch = _keyData.btn.components.L1;


    // if(_keyData.btn.components.A > 0){
    //     mode = 0;
    //     printf("================ mode: %d ==================", mode);
    // } else if(_keyData.btn.components.B > 0){
    //     mode = 1;
    //     printf("================ mode: %d ==================", mode);
    // }else if(_keyData.btn.components.X > 0){
    //     mode = 2;
    //     printf("================ mode: %d ==================", mode);
    // }else if(_keyData.btn.components.Y > 0){
    //     mode = 3;
    //     printf("================ mode: %d ==================", mode);
    // }else if(_keyData.btn.components.up > 0){
    //     mode = 4;
    //     printf("================ mode: %d ==================", mode);
    // }else if(_keyData.btn.components.right > 0){
    //     mode = 5;
    //     printf("================ mode: %d ==================", mode);
    // }else if(_keyData.btn.components.down > 0){
    //     mode = 6;
    //     printf("================ mode: %d ==================", mode);
    // }else if(_keyData.btn.components.left > 0){
    //     mode = 7;
    //     printf("================ mode: %d ==================", mode);
    // }else{
    //     mode = -1;
    // }

    if(_keyData.btn.components.A > 0){
        mode = 0;
        for (int i = 0; i<7; i++){
            ARX_ARM.target_pos[i] = ARX_ARM.current_pos[i];
            ARX_ARM.target_pos_temp[i] = ARX_ARM.current_pos[i];
        }
        printf("================ stop ==================");
        stop_flag = true;
    } else if(_keyData.btn.components.B > 0){
        mode = 1;
        stop_flag = false;
        printf("================ moving ==================");

    //     // 7 for changing commands 
    //     joint_state.q[7] += 1.;
    //     if (joint_state.q[7] >= 3){
    //         joint_state.q[7] = 0;
    } else if(_keyData.btn.components.X > 0){
    	if (stop_flag) {
	   mode = 2;
	   printf("================ gripper open ===========");
	   ARX_ARM.target_pos[6] += 0.25;
	}
    } else if(_keyData.btn.components.Y > 0){
    	if (stop_flag) {
	   mode = 3;
	   printf("=============== gripper close ===========");
	   ARX_ARM.target_pos[6] -= 0.25;
	}
    }

    // }else if(_keyData.btn.components.X > 0){
    //     mode = 2;
        
    //     // TODO：这里还需要考虑电机发热的问题
    //     if (stop_flag){
    //         // 6 for gripper target angle
    //          ARX_ARM.target_pos[6] = 0.5;
    //     }
    // }else if(_keyData.btn.components.Y > 0){
    //     mode = 3;
        
    //     if (stop_flag){
    //         // 6 for gripper target angle
    //          ARX_ARM.target_pos[6] = 0;
    //     }
    // }else if(_keyData.btn.components.up > 0){
    //     mode = 4;
    // }else if(_keyData.btn.components.right > 0){
    //     mode = 5;
    // }else if(_keyData.btn.components.down > 0){
    //     mode = 6;
    // }else if(_keyData.btn.components.left > 0){
    //     mode = 7;
    // }else{
    //     mode = -1;
    // }

    rc_command.mode = mode;

    // publish state to LCM
    for(int i = 0; i < 12; i++){
        joint_state.q[i] = state.motorState[i].q;
        joint_state.qd[i] = state.motorState[i].dq;
        joint_state.tau_est[i] = state.motorState[i].tauEst;
        std::cout << "state[" << i <<"]: " << state.motorState[i].q << std::endl;

    }
    for(int i = 0; i < 4; i++){
        body_state_simple.quat[i] = state.imu.quaternion[i];
    }
    for(int i = 0; i < 3; i++){
        body_state_simple.rpy[i] = state.imu.rpy[i];
        body_state_simple.aBody[i] = state.imu.accelerometer[i];
        body_state_simple.omegaBody[i] = state.imu.gyroscope[i];
    }
    // printf("========= roll: %f =========\n", body_state_simple.rpy[0]);
    // printf("========= pitch: %f =========\n", body_state_simple.rpy[1]);
    // printf("========= yaw: %f =========\n", body_state_simple.rpy[2]);



    for(int i = 0; i < 4; i++){
        body_state_simple.contact_estimate[i] = state.footForce[i];
    }

    // arm current joint pose
    ARX_ARM.get_joint();
    for (int i = 0; i < 6; i++){
        joint_state.q_arm[i] = ARX_ARM.current_pos[i];
        // std::cout << "机械臂位置状态：arm_joint_states[" << i <<"]: " << ARX_ARM.current_pos[i] << std::endl;
    }
    std::cout << std::endl;


    // if (key == 's'){
    //     stop_flag = true;
    //     for (int i = 0; i<7; i++){
    //         ARX_ARM.target_pos[i] = ARX_ARM.current_pos[i];
    //         ARX_ARM.target_pos_temp[i] = ARX_ARM.current_pos[i];
    //     }
    //     printf("================ stop ==================");
    
    // }
    // else if (key == 'm'){
    //     printf("================ moving ==================");
    //     stop_flag = false;
    //     joint_state.q_arm[7] += 1.;
    //     if (joint_state.q_arm[7] >= 3){
    //         joint_state.q_arm[7] = 0;
    //     }
    // }

    _simpleLCM.publish("leg_state_estimator_data", &body_state_simple);
    _simpleLCM.publish("leg_control_data", &joint_state);
    _simpleLCM.publish("rc_command", &rc_command);

    /////////////////////////////////////////////////////////////////////
    // 上面发送状态
    // 下面控制底层
    /////////////////////////////////////////////////////////////////////

    if(_firstRun && joint_state.q[0] != 0){
        for(int i = 0; i < 12; i++){
            joint_command.q_des[i] = joint_state.q[i];
        }

        _firstRun = false;
    }

    for(int i = 0; i < 12; i++){
        cmd.motorCmd[i].q = joint_command.q_des[i];
        cmd.motorCmd[i].dq = joint_command.qd_des[i];
        cmd.motorCmd[i].Kp = joint_command.kp[i];
        cmd.motorCmd[i].Kd = joint_command.kd[i];
        cmd.motorCmd[i].tau = joint_command.tau_ff[i];
    }


    // control dog
    safe.PositionLimit(cmd);
    int res1 = safe.PowerProtect(cmd, state, 9);
    udp.SetSend(cmd);



    if (!stop_flag){
        for(int i = 0; i < 6; i++){  // 只是控制6个关节
            ARX_ARM.target_pos[i] =  joint_command.q_arm_des[i];
     
            // std::cout << "机械臂控制指令：arm_joint_command[" << i <<"]: " << ARX_ARM.target_pos[i] << std::endl;
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

    usleep(4200);
}


int main(void)
{
    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    Custom custom(LOWLEVEL);
    custom.init();
    // InitEnvironment();
    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    // loop_udpSend.start();
    loop_udpRecv.start();
    loop_udpSend.start();
    loop_control.start();



    while(1){
        sleep(10);
    };

    loop_control.shutdown();
    loop_udpSend.shutdown();
    loop_udpRecv.shutdown();

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

    return 0;
}
