# 将文件传到 lower control board
```
cd go1_gym_deploy/scripts && ./send_to_unitree.sh
```
</br>
</br>

# 如果是第一次需要下载和安装 docker, 只需要一次就行

```
ssh unitree@192.168.123.15
```

```
chmod +x installer/install_deployment_code.sh
cd ~/go1_gym/go1_gym_deploy/scripts
sudo ../installer/install_deployment_code.sh
```
</br>
</br>

# Running the Controller  <a name="runcontroller"></a>

Place the robot into damping mode. The control sequence is: [L2+A], [L2+B], [L1+L2+START]. After this, the robot should sit on the ground and the joints should move freely. 

Now, ssh to `unitree@192.168.123.15` and run the following two commands to start the controller. <b>This will operate the robot in low-level control mode. Make sure your Go1 is hung up.</b>

First:
```
# 需要改成对应的rule串口号
cd ~/go1_gym/go1_gym_deploy/unitree_legged_sdk_bin
sh setup_rule.sh
sh reopen.sh
sh reopen.sh
sh make.sh
./lcm_position
```

Second:
```
cd ~/go1_gym/go1_gym_deploy/docker
sudo make autostart
```
