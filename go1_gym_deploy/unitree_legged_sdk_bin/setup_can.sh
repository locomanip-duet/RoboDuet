#!/bin/bash

# Set up CAN0
while [ 1 ]
do
sudo ip link set up can0 type can bitrate 1000000
sleep 1
done
