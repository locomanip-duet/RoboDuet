#!/bin/bash
sudo cp arx_can.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo -S slcand -o -f -s8 /dev/canable0 can0
sudo ifconfig can0 up


