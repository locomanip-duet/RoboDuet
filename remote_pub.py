#!/usr/bin/env python
import lcm
from time import sleep
import signal
import sys
import numpy as np
import zmq
import pickle

sys.path.append("../")

from go1_gym_deploy.lcm_types.arm_actions_t import arm_actions_t

 
# NOTE：此IP:PORT 为 VR 端串流主机用于发送消息的端口
# 在 jetson 上的订阅ip/pull ip 也设置成这个主机的 ip+port
GLOBAL_IP = "192.168.1.134"
GLOBAL_PORT = "34565"
 

def calibrate_dual_arx(sock: zmq.Socket, start_pose=None):
    print("start to calibrate,don't move the controller!")
    count = np.zeros(14)
    for i in range(50):
        action_bin = sock.recv()
        action_ = pickle.loads(action_bin)
        action_ = np.array(action_)
        count += action_[:14]
    print("calibration done")
    count = count / 50

    roll1 = -count[2]
    pitch1 = -count[0]
    yaw1 = count[1]
    x1 = -count[5]
    y1 = -count[3]
    z1 = count[4]

    roll2 = -count[9]
    pitch2 = -count[7]
    yaw2 = count[8]
    x2 = -count[12]
    y2 = -count[10]
    z2 = count[11]


    if start_pose is None:
        return np.array([-x1, -y1, -z1, -roll1, -pitch1, -yaw1,  0.,
                         -x2, -y2, -z2, -roll2, -pitch2, -yaw2, 0.])
    else:
        return np.array([-x1, -y1, -z1, -roll1, -pitch1, -yaw1,  0.,
                         -x2, -y2, -z2, -roll2, -pitch2, -yaw2, 0.]) + start_pose

def master_to_arm_arx(action, offset):
    roll = -action[2]
    pitch = -action[0]
    yaw = action[1]
    x = -action[5]
    y = -action[3]
    z = action[4]
    gripper = action[6]

    roll1 = -action[9]
    pitch1 = -action[7]
    yaw1 = action[8]
    x1 = -action[12]
    y1 = -action[10]
    z1 = action[11]
    gripper1 = action[13]
    # print("gripper: ", gripper)
    # print("gripper1: ", gripper1)
    master_action = np.array([x, y, z, roll, pitch, yaw, gripper,
                              x1, y1, z1, roll1, pitch1, yaw1, gripper1])
    
    formatted_tuple = tuple(f"{num:.4f}" for num in master_action+offset)
    # print("action: ", formatted_tuple)
    # return master_action + offset
    return master_action+offset, action[14], action[15], action[16], action[17], action[18], action[19]


def get_master_action_arx(sock: zmq.Socket, offset):
    action_bin = sock.recv()
    action_ = pickle.loads(action_bin)
    action, a, b, x, y, thumb_x, thumb_y = master_to_arm_arx(action_, offset)
    return action, a, b, x, y, thumb_x, thumb_y


def test():
    np.set_printoptions(precision=3)
    
    lcm_node = lcm.LCM("udpm://239.255.76.67:7136?ttl=255")

    arm_pose_msg = arm_actions_t()

    start_pos = np.array([0,0,0,0,0,0,1000])


    #     paper_box_width = 800
    # plastic_box_width = 680
    
    previous_pose = np.zeros(6)
    arm_delta = 0.01


    print("wait for zmq socket connect")
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.connect(f"tcp://{GLOBAL_IP}:{GLOBAL_PORT}")
    # sock = SocketServer(ip_port=('192.168.1.121', 34564))
    print("socket connected!")
    
    # 将资源关闭信号处理函数定义在 socket 初始化之后，安全关闭上下文   
    shutdown = False
    def signal_handler(sig, frame):
        global shutdown
        shutdown = True
        sock.close()
        context.term()
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    
    print("press A to start calibration")
    while not shutdown:
        while not shutdown:
            _,a,_,_,_,_,_ = get_master_action_arx(sock, np.zeros(14))
            if a == 1:
                offset = calibrate_dual_arx(sock)
                break

        print("press B to stop teleopration, then press X to reset or press A to restart")
        while not shutdown:
            action, a, b, x, y, thumb_x, thumb_y = get_master_action_arx(sock, offset)
            if b == 1:
                while not shutdown:
                    action, a, b, x, y, _,_ = get_master_action_arx(sock, offset)
                    if x == 1:
                        arm_pose_msg.data = start_pos[0:6]
                        
                        lcm_node.publish("arm_control_data", arm_pose_msg.encode())

                        print("x")

                        return 0

                    if a == 1:
                        arm_pose_msg.data = start_pos[0:6]
                        
                        print("a")

                        lcm_node.publish("arm_control_data", arm_pose_msg.encode())

                        break
                break


            if np.sum(np.abs(previous_pose[0:3] - action[0:3])+0.2*np.abs(previous_pose[3:6] - action[3:6])) > arm_delta:
                arm_pose_msg.data = action[0:6]
                previous_pose = action[0:6]
                print('left', arm_pose_msg.data)
                lcm_node.publish("arm_control_data", arm_pose_msg.encode())


            sleep(0.001)




if __name__ == '__main__':
    test()
