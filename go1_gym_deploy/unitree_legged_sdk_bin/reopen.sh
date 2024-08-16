sudo udevadm control --reload-rules && sudo udevadm trigger
sudo slcand -o -f -s8 /dev/canable0 can0
sudo ifconfig can0 up
