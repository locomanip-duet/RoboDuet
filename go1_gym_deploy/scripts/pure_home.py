from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_modules.core import InterbotixRobotXSCore
bot = InterbotixManipulatorXS("wx250s", "arm", "gripper")
core = InterbotixRobotXSCore(robot_model='wx250s' ,robot_name="wx250s")
bot.arm.go_to_home_pose()