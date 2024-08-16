#include "App/arm_control.h"

extern OD_Motor_Msg rv_motor_msg[10];

extern float magic_pos[3];
extern float magic_angle[3];
extern int magic_switch[2];
//斜坡函数
float arx_arm::ramp(float goal, float current, float ramp_k)
{
    float retval = 0.0f;
    float delta = 0.0f;
    delta = goal - current;
    if (delta > 0)
    {
        if (delta > ramp_k)
        {  
                current += ramp_k;
        }   
        else
        {
            current += delta;
        }
    }
    else
    {
        if (delta < -ramp_k)
        {
                current += -ramp_k;
        }
        else
        {
                current += delta;
        }
    }	
    retval = current;
    return retval;
}

arx_arm::arx_arm()
{
    CAN_Handlej.Send_moto_Cmd1(1, 0, 0, 0, 0, 0);
    usleep(1000);
    CAN_Handlej.Send_moto_Cmd1(2, 0, 0, 0, 0, 0);
    usleep(1000);
    CAN_Handlej.Send_moto_Cmd1(4, 0, 0, 0, 0, 0);
    usleep(1000);
    CAN_Handlej.Enable_Moto(0x05);
    usleep(1000);
    CAN_Handlej.Enable_Moto(0x06);
    usleep(1000);
    CAN_Handlej.Enable_Moto(0x07);
    usleep(1000);
    CAN_Handlej.Enable_Moto(0x08);
    usleep(1000);
}

void arx_arm::get_joint()
{
    current_pos[0] = rv_motor_msg[0].angle_actual_rad;
    current_pos[1] = rv_motor_msg[1].angle_actual_rad;
    current_pos[2] = rv_motor_msg[3].angle_actual_rad;
    current_pos[3] = rv_motor_msg[4].angle_actual_rad;
    current_pos[4] = rv_motor_msg[5].angle_actual_rad;
    current_pos[5] = rv_motor_msg[6].angle_actual_rad;
    current_pos[6] = rv_motor_msg[7].angle_actual_rad;
    printf("\033[32mangle_ ros_joint = 1>%f 2>%f 3>%f 4>%f 5>%f 6>%f 7>%f \r\n\033[0m",  current_pos[0], \
                                                                                         current_pos[1], \
                                                                                         current_pos[2], \
                                                                                         current_pos[3], \
                                                                                         current_pos[4], \
                                                                                         current_pos[5], \
                                                                                         current_pos[6]);

    current_vel[0] = rv_motor_msg[0].speed_actual_rad;
    current_vel[1] = rv_motor_msg[1].speed_actual_rad;
    current_vel[2] = rv_motor_msg[3].speed_actual_rad;
    current_vel[3] = rv_motor_msg[4].speed_actual_rad;
    current_vel[4] = rv_motor_msg[5].speed_actual_rad;
    current_vel[5] = rv_motor_msg[6].speed_actual_rad;
    current_vel[6] = rv_motor_msg[7].speed_actual_rad;

    // printf("\033[32mcurrent_vel = 1>%f 2>%f 3>%f 4>%f 5>%f 6>%f 7>%f \r\n\033[0m",  current_vel[0], \
    //                                                                                      current_vel[1], \
    //                                                                                      current_vel[2], \
    //                                                                                      current_vel[3], \
    //                                                                                      current_vel[4], \
    //                                                                                      current_vel[5], \
    //                                                                                      current_vel[6]);

    for(int i=0;i<7;i++)
    {
        if(current_pos[i]==0)
        {
        printf("motor %d is not connected\r\n",i+1);
        }
        
    }

    current_torque[0] = rv_motor_msg[0].current_actual_float;
    current_torque[1] = rv_motor_msg[1].current_actual_float;
    current_torque[2] = rv_motor_msg[3].current_actual_float;
    current_torque[3] = rv_motor_msg[4].current_actual_float;
    current_torque[4] = rv_motor_msg[5].current_actual_float;
    current_torque[5] = rv_motor_msg[6].current_actual_float;
    current_torque[6] = rv_motor_msg[7].current_actual_float;

    float set_max_torque = 9;
    
        for (float num : current_torque) {
            if (abs(num) > set_max_torque) {
                temp_current_normal++;
            }
        }

        for (float num : current_torque) {
            if(abs(num) > set_max_torque)
            {
                temp_condition = false;
            }
        }
        if(temp_condition)
        {
            temp_current_normal=0;
        }

        if(temp_current_normal>100)
        {
            current_normal = false;
        }

}


void arx_arm::update_real()
{
        motor_control();
        //joint_control();
        // gripper_control(1.0f);//1.0f是无用的，测试专用
    return;
}

void arx_arm::motor_control()
{
    if(current_normal)
    {

        limit_joint(target_pos);

        target_pos_temp[0] = ramp(target_pos[0], target_pos_temp[0], 0.005);
        target_pos_temp[1] = ramp(target_pos[1], target_pos_temp[1], 0.005);
        target_pos_temp[2] = ramp(target_pos[2], target_pos_temp[2], 0.005);
        target_pos_temp[3] = ramp(target_pos[3], target_pos_temp[3], 0.005);
        target_pos_temp[4] = ramp(target_pos[4], target_pos_temp[4], 0.005);
        target_pos_temp[5] = ramp(target_pos[5], target_pos_temp[5], 0.005);
        target_pos_temp[6] = ramp(target_pos[6], target_pos_temp[6], 0.005);

        // CAN_Handlej.Send_moto_Cmd1(1, 90, 28, target_pos_temp[0], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd1(2, 110, 36, target_pos_temp[1], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd1(4, 150, 50, target_pos_temp[2], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(5, 30, 2, target_pos_temp[3], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(6, 25, 1, target_pos_temp[4], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(7, 20, 1,   target_pos_temp[5], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(8, 5, 1,   target_pos_temp[6], 0, 0);usleep(200);

        CAN_Handlej.Send_moto_Cmd1(1, 90, 28, target_pos_temp[0], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd1(2, 60, 40, target_pos_temp[1], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd1(4, 130, 36, target_pos_temp[2], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(5, 30, 1.2, target_pos_temp[3], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(6, 25, 1, target_pos_temp[4], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(7, 20, 1,   target_pos_temp[5], 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(8, 5, 1,   target_pos_temp[6], 0, 0);usleep(200);

        // limit_joint(target_pos);

        // // target_pos_temp[0] =ramp(target_pos[0],target_pos_temp[0],0.0001);
        // // target_pos_temp[1] =ramp(target_pos[1],target_pos_temp[1],0.0001);
        // // target_pos_temp[2] =ramp(target_pos[2],target_pos_temp[2],0.0001);
        // // target_pos_temp[3] =ramp(target_pos[3],target_pos_temp[3],0.0001);
        // // target_pos_temp[4] =ramp(target_pos[4],target_pos_temp[4],0.0001);
        // // target_pos_temp[5] =ramp(target_pos[5],target_pos_temp[5],0.0001);
        // // target_pos_temp[6] =ramp(target_pos[6],target_pos_temp[6],0.0001);

        // float max_delta = -1000;
        // float thres = 0.001;
        // float delta[7] = {};
        // for(int i=0; i<7; i++){
        //     delta[i] = target_pos[i] - target_pos_temp[i];
        //     if(std::abs(delta[i]) > max_delta) max_delta = std::abs(delta[i]);
        // }
        // if (max_delta > thres){
        //     for(int i=0; i<7; i++){
        //         target_pos_temp[i] += thres * delta[i] / max_delta;
        //     }
        // }
        // else {
        //     for(int i=0; i<7; i++){
        //         target_pos_temp[i] += delta[i];
        //     }
        // }
        // // for(int i=0; i<7; i++){

        // //     if (delta[i] > thres){

        // //         target_pos_temp[i] += thres * delta[i] / max_delta;
        // //     }
        // //     else if (delta[i] < -thres){
        // //         target_pos_temp[i] += thres * delta[i] / max_delta;
        // //     }
        // //     else{
        // //         target_pos_temp[i] += delta[i];
        // //     }
        // // }

        // // target_pos_temp[0] += (target_pos[0] - target_pos_temp[0]) * 0.0001;
        // // target_pos_temp[1] += (target_pos[1] - target_pos_temp[1]) * 0.0001;
        // // target_pos_temp[2] += (target_pos[2] - target_pos_temp[2]) * 0.0001;
        // // target_pos_temp[3] += (target_pos[3] - target_pos_temp[3]) * 0.0001;
        // // target_pos_temp[4] += (target_pos[4] - target_pos_temp[4]) * 0.0001;
        // // target_pos_temp[5] += (target_pos[5] - target_pos_temp[5]) * 0.0001;
        // // target_pos_temp[6] += (target_pos[6] - target_pos_temp[6]) * 0.0001;

        // // printf("\033[32m delta = 1>%f 2>%f 3>%f 4>%f 5>%f 6>%f 7>%f \r\n\033[0m",  target_pos_temp[0], \
        // //                                                                                      target_pos_temp[1], \
        // //                                                                                      target_pos_temp[2], \
        // //                                                                                      target_pos_temp[3], \
        // //                                                                                      target_pos_temp[4], \
        // //                                                                                      target_pos_temp[5], \
        // //                                                                                      target_pos_temp[6]);

        // // CAN_Handlej.Send_moto_Cmd1(1, 150, 12, target_pos_temp[0], 0, 0);usleep(200);
        // // CAN_Handlej.Send_moto_Cmd1(2, 150, 12, target_pos_temp[1], 0, 0);usleep(200);
        // // CAN_Handlej.Send_moto_Cmd1(4, 150, 12, target_pos_temp[2], 0, 0);usleep(200);
        // // CAN_Handlej.Send_moto_Cmd2(5, 30, 0.8, target_pos_temp[3], 0, 0);usleep(200);
        // // CAN_Handlej.Send_moto_Cmd2(6, 25, 0.8, target_pos_temp[4], 0, 0);usleep(200);
        // // CAN_Handlej.Send_moto_Cmd2(7, 10, 1,   target_pos_temp[5], 0, 0);usleep(200);
        // // CAN_Handlej.Send_moto_Cmd2(8, 10, 1,   target_pos_temp[6], 0, 0);usleep(200);

        // float torque[7]={0};
        // float p_gains[7] = {15, 10 ,20, 5, 5, 5, 5};
        // float d_gains[7] = {5, 6 ,4, 1, 1, 1, 1};

        // printf("torque: ");
        // for(int i =0 ;i < 7; i++){
        //     torque[i] =  p_gains[i] * (target_pos_temp[i] - current_pos[i]) - d_gains[i] * current_vel[i];
        //     printf(" %f  ", torque[i]);
        // }
        // printf("\n");

        // // printf("p_fen: ");
        // // for(int i =0 ;i < 7; i++){
        // //     printf(" %f  ", p_gains[i] * (target_pos_temp[i] - current_pos[i]));
        // // }
        // // printf("\n");

        // printf("d_fen: ");
        // for(int i =0 ;i < 7; i++){
        //     printf(" %f  ", - d_gains[i] * current_vel[i]);
        // }
        // printf("\n");

        // // printf("p_delta: ");
        // // for(int i =0 ;i < 7; i++){
        // //     printf(" %f  ", target_pos_temp[i] - current_pos[i]);
        // // }
        // // printf("\n");

        // // printf("target_delta: ");
        // // for(int i =0 ;i < 7; i++){
        // //     printf(" %f  ", target_pos[i] - target_pos_temp[i]);
        // // }
        // // printf("\n");

        // printf("target_current: ");
        // for(int i =0 ;i < 7; i++){
        //     printf(" %f  ", target_pos[i] - current_pos[i]);
        // }
        // printf("\n");

        // printf("\n");
        // printf("\n");
        
        // CAN_Handlej.Send_moto_Cmd1(1, 15, 5, target_pos_temp[0], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd1(2, 150, 12, target_pos_temp[1], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd1(4, 150, 12, target_pos_temp[2], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(5, 5, 1, target_pos_temp[3], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(6, 5, 1, target_pos_temp[4], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(7, 5, 1,   target_pos_temp[5], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(8, 5, 1,   target_pos_temp[6], 0, 0);usleep(200);


        // CAN_Handlej.Send_moto_Cmd1(1, 100,  10, target_pos_temp[0], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd1(2, 150,  10, target_pos_temp[1], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd1(4, 20, 2, target_pos_temp[2], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(5, 5,  1, target_pos_temp[3], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(6, 5,  1, target_pos_temp[4], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(7, 5,  1, target_pos_temp[5], 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(8, 5,  1, target_pos_temp[6], 0, 0);usleep(200);

        // CAN_Handlej.Send_moto_Cmd1(1, 0, 12, 0, 0, 0);
        // CAN_Handlej.Send_moto_Cmd1(2, 0, 12, 0, 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd1(4, 0, 12, 0, 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(5, 0, 1, 0, 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(6, 0, 1, 0, 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(7, 0, 1, 0, 0, 0);usleep(200);
        // CAN_Handlej.Send_moto_Cmd2(8, 0, 1, 0, 0, 0);usleep(200);


    }else
    {
        CAN_Handlej.Send_moto_Cmd1(1, 0, 12, 0, 0, 0);
        CAN_Handlej.Send_moto_Cmd1(2, 0, 12, 0, 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd1(4, 0, 12, 0, 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(5, 0, 1, 0, 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(6, 0, 1, 0, 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(7, 0, 1, 0, 0, 0);usleep(200);
        CAN_Handlej.Send_moto_Cmd2(8, 0, 1, 0, 0, 0);usleep(200);
        printf("current warning !safe mode!!!!!!!!!\r\n");
    }
}

void arx_arm::joint_control()
{

}

//调用函数:R键复位，开局复位
void arx_arm::init_step()
{

}

void arx_arm::gripper_control(float gripper_pos)
{

}

void arx_arm::limit_joint(float* Set_Pos)
{
    Set_Pos[0] = limit<float>(Set_Pos[0], Lower_Joint[0], Upper_Joint[0]);
    Set_Pos[1] = limit<float>(Set_Pos[1], Lower_Joint[1], Upper_Joint[1]);
    Set_Pos[2] = limit<float>(Set_Pos[2], Lower_Joint[2], Upper_Joint[2]);
    Set_Pos[3] = limit<float>(Set_Pos[3], Lower_Joint[3], Upper_Joint[3]);
    Set_Pos[4] = limit<float>(Set_Pos[4], Lower_Joint[4], Upper_Joint[4]);
    Set_Pos[5] = limit<float>(Set_Pos[5], Lower_Joint[5], Upper_Joint[5]);
    Set_Pos[6] = limit<float>(Set_Pos[6], Lower_Joint[6], Upper_Joint[6]);


}