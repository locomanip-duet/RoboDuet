# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build

# Include any dependencies generated for this target.
include CMakeFiles/lcm_position_arm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lcm_position_arm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lcm_position_arm.dir/flags.make

CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o: CMakeFiles/lcm_position_arm.dir/flags.make
CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o: ../lcm_position_arm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o -c /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/lcm_position_arm.cpp

CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/lcm_position_arm.cpp > CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.i

CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/lcm_position_arm.cpp -o CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.s

CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o.requires:

.PHONY : CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o.requires

CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o.provides: CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o.requires
	$(MAKE) -f CMakeFiles/lcm_position_arm.dir/build.make CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o.provides.build
.PHONY : CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o.provides

CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o.provides.build: CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o


CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o: CMakeFiles/lcm_position_arm.dir/flags.make
CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o: ../arx/src/utility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o -c /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/utility.cpp

CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/utility.cpp > CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.i

CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/utility.cpp -o CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.s

CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o.requires:

.PHONY : CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o.requires

CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o.provides: CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o.requires
	$(MAKE) -f CMakeFiles/lcm_position_arm.dir/build.make CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o.provides.build
.PHONY : CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o.provides

CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o.provides.build: CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o


CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o: CMakeFiles/lcm_position_arm.dir/flags.make
CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o: ../arx/src/Hardware/motor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o -c /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/Hardware/motor.cpp

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/Hardware/motor.cpp > CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.i

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/Hardware/motor.cpp -o CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.s

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o.requires:

.PHONY : CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o.requires

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o.provides: CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o.requires
	$(MAKE) -f CMakeFiles/lcm_position_arm.dir/build.make CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o.provides.build
.PHONY : CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o.provides

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o.provides.build: CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o


CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o: CMakeFiles/lcm_position_arm.dir/flags.make
CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o: ../arx/src/Hardware/math_ops.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o -c /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/Hardware/math_ops.cpp

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/Hardware/math_ops.cpp > CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.i

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/Hardware/math_ops.cpp -o CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.s

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o.requires:

.PHONY : CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o.requires

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o.provides: CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o.requires
	$(MAKE) -f CMakeFiles/lcm_position_arm.dir/build.make CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o.provides.build
.PHONY : CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o.provides

CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o.provides.build: CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o


CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o: CMakeFiles/lcm_position_arm.dir/flags.make
CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o: ../arx/src/App/arm_control.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o -c /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/App/arm_control.cpp

CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/App/arm_control.cpp > CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.i

CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/arx/src/App/arm_control.cpp -o CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.s

CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o.requires:

.PHONY : CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o.requires

CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o.provides: CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o.requires
	$(MAKE) -f CMakeFiles/lcm_position_arm.dir/build.make CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o.provides.build
.PHONY : CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o.provides

CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o.provides.build: CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o


# Object files for target lcm_position_arm
lcm_position_arm_OBJECTS = \
"CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o" \
"CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o" \
"CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o" \
"CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o" \
"CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o"

# External object files for target lcm_position_arm
lcm_position_arm_EXTERNAL_OBJECTS =

../lcm_position_arm: CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o
../lcm_position_arm: CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o
../lcm_position_arm: CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o
../lcm_position_arm: CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o
../lcm_position_arm: CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o
../lcm_position_arm: CMakeFiles/lcm_position_arm.dir/build.make
../lcm_position_arm: ../arx/libcan/arm64/libarmcan.so
../lcm_position_arm: ../arx/libcan/arm64/libarmscan.so
../lcm_position_arm: ../arx/libcan/arm64/libarmarx_s.so
../lcm_position_arm: ../arx/libcan/arm64/libarmkey.so
../lcm_position_arm: CMakeFiles/lcm_position_arm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable ../lcm_position_arm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lcm_position_arm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lcm_position_arm.dir/build: ../lcm_position_arm

.PHONY : CMakeFiles/lcm_position_arm.dir/build

CMakeFiles/lcm_position_arm.dir/requires: CMakeFiles/lcm_position_arm.dir/lcm_position_arm.cpp.o.requires
CMakeFiles/lcm_position_arm.dir/requires: CMakeFiles/lcm_position_arm.dir/arx/src/utility.cpp.o.requires
CMakeFiles/lcm_position_arm.dir/requires: CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/motor.cpp.o.requires
CMakeFiles/lcm_position_arm.dir/requires: CMakeFiles/lcm_position_arm.dir/arx/src/Hardware/math_ops.cpp.o.requires
CMakeFiles/lcm_position_arm.dir/requires: CMakeFiles/lcm_position_arm.dir/arx/src/App/arm_control.cpp.o.requires

.PHONY : CMakeFiles/lcm_position_arm.dir/requires

CMakeFiles/lcm_position_arm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lcm_position_arm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lcm_position_arm.dir/clean

CMakeFiles/lcm_position_arm.dir/depend:
	cd /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build /home/unitree/policy_deployment/go1_gym_deploy/unitree_legged_sdk_bin/build/CMakeFiles/lcm_position_arm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lcm_position_arm.dir/depend

