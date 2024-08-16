sudo rm -rf build
rm lcm_position
rm lcm_position_arm
mkdir build
cd build
cmake ..
make
cd ..
