
import isaacgym
assert isaacgym
import torch
import argparse

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset

import wandb
import os
import os.path as osp
from datetime import datetime
from pathlib import Path
from go1_gym import MINI_GYM_ROOT_DIR
import shutil
import pickle

from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv

from go1_gym_learn.ppo_cse_automatic import Runner
from go1_gym_learn.ppo_cse_automatic.ppo import PPO_Args
from go1_gym_learn.ppo_cse_automatic import RunnerArgs
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogAC_Args
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmAC_Args


from go1_gym.utils import format_code, set_seed, global_switch

def train_go1(headless=True):
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)
    
    Cfg.commands.distributional_commands = False
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "P"
    Cfg.domain_rand.added_mass_range = [-2.0, 2.0]
    Cfg.env.observe_two_prev_actions = False
    Cfg.commands.body_roll_range = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.4, 0.4]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]

    Cfg.env.num_envs = args.num_envs

    Cfg.env.keep_arm_fixed = True
    
    Cfg.terrain.mesh_type = "plane"
    if Cfg.terrain.mesh_type == "plane":
        Cfg.terrain.teleport_robots = False
    Cfg.control.update_obs_freq = 50 # Hz
    Cfg.env.num_actions = 18
    Cfg.env.num_observations = 63
    
    Cfg.hybrid.reward_scales.tracking_lin_vel = 0.5 * Cfg.reward_scales.tracking_lin_vel
    Cfg.hybrid.reward_scales.tracking_ang_vel = 0.5 * Cfg.reward_scales.tracking_ang_vel
    Cfg.reward_scales.arm_energy = -0.00004
    Cfg.reward_scales.loco_energy = -0.00004

    Cfg.reward_scales.jump = -0.00
    Cfg.rewards.terminal_body_height = 0.28
    Cfg.rewards.use_terminal_body_height = True
    global_switch.pretrained_to_hybrid_start = 10000  # 2000 with pretrained, 10000 from scratch
    global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 5000
    
    Cfg.env.priv_observe_vel = False
    Cfg.commands.global_reference = False
    Cfg.env.priv_observe_high_freq_goal = False
    Cfg.dog.dog_num_privileged_obs = 2
    Cfg.arm.arm_num_privileged_obs = 9
    Cfg.env.num_privileged_obs = 9
    
    Cfg.hybrid.plan_vel = True
    if Cfg.hybrid.plan_vel:
        Cfg.arm.num_actions_arm_cd += 2
        Cfg.hybrid.reward_scales.arm_vel_control = -1

    
    now = datetime.now()
    stem = Path(__file__).stem
    wandb.init(project="controller",
               group=args.run_name,
               entity="dwb",
               mode=mode,
               notes=args.notes,
               name=f'{now.strftime("%Y-%m-%d")}/{stem}/{now.strftime("%H%M%S.%f")}',
               tags=args.tags,
               dir=f"{MINI_GYM_ROOT_DIR}")
    
    args.log_dir = osp.join(f"{MINI_GYM_ROOT_DIR}/runs/{args.run_name}", wandb.run.name)
    args.log_dir += f'_seed{args.seed}'
    if not args.debug:
        
        
        # Example:
        # >>> # run_log_dir
        # >>> /home/pgp/walk-these-ways/runs/wandb/run-20231231_021709-zdwnpnlq/files/2023-12-31/train_ppo/021707.917935
        # >>> # wandb.run.dir
        # >>> /home/pgp/walk-these-ways/runs/wandb/run-20231231_021709-zdwnpnlq/files
        # >>> # wandb.run.name
        # >>> 2023-12-31/train_ppo/021707.917935
        run_log_dir = osp.join(wandb.run.dir, wandb.run.name)


        #  for example: /home/pi7113t/dog/dwb-wtw/runs/wo_any_random/2023-12-31/train_ppo/021707.917935
        os.makedirs(osp.join(args.log_dir, "checkpoints_arm"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "checkpoints_dog"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "scripts"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "videos"), exist_ok=True)
        os.makedirs(f"{MINI_GYM_ROOT_DIR}/tmp/deploy_model", exist_ok=True)

        # 将版本的 commit 代码 保存到 runs
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/scripts/auto_train.py", f"{args.log_dir}/scripts/auto_train.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/legged_robot.py", f"{args.log_dir}/scripts/legged_robot.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/legged_robot_config.py", f"{args.log_dir}/scripts/legged_robot_config.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/__init__.py", f"{args.log_dir}/scripts/env__init__.py")
        
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/__init__.py", f"{args.log_dir}/scripts/ppo_cse_automatic__init__.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/arm_ac.py", f"{args.log_dir}/scripts/arm_ac.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/dog_ac.py", f"{args.log_dir}/scripts/dog_ac.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/ppo.py", f"{args.log_dir}/scripts/ppo.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/rollout_storage.py", f"{args.log_dir}/scripts/rollout_storage.py")
        
        
        # # 将版本的 commit 代码 保存到 wandb
        os.makedirs(f"{wandb.run.dir}/scripts", exist_ok=True)
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/scripts/auto_train.py", f"{wandb.run.dir}/scripts/auto_train.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/legged_robot.py", f"{wandb.run.dir}/scripts/legged_robot.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/legged_robot_config.py", f"{wandb.run.dir}/scripts/legged_robot_config.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/__init__.py", f"{wandb.run.dir}/scripts/env__init__.py")
        
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/__init__.py", f"{wandb.run.dir}/scripts/ppo_cse_automatic__init__.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/arm_ac.py", f"{wandb.run.dir}/scripts/arm_ac.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/dog_ac.py", f"{wandb.run.dir}/scripts/dog_ac.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/ppo.py", f"{wandb.run.dir}/scripts/ppo.py")
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/rollout_storage.py", f"{wandb.run.dir}/scripts/rollout_storage.py")
        
        
        # 将参数保存到 runs
        temp_dict = {"Cfg": vars(Cfg), "RunnerArgs": vars(RunnerArgs), "ArmAC_Args": vars(ArmAC_Args), "DogAC_Args": vars(DogAC_Args), "PPO_Args": vars(PPO_Args),}

        with open(f"{args.log_dir}/params.txt", "w", encoding="utf-8") as f:
            format_temp_dict = format_code(str(temp_dict))
            f.write(format_temp_dict)
        # 将参数保存到 wandb
        # shutil.copyfile(f"{args.log_dir}/params.txt", f"{wandb.run.dir}/scripts_commit/params.txt")

        with open(osp.join(args.log_dir, "parameters.pkl"), 'wb') as f:
            pickle.dump(temp_dict, f)
        wandb.save(osp.join(args.log_dir, "parameters.pkl"), policy="now")
        
        # wandb.save(f"{args.log_dir}/params.txt", policy="now")
        # wandb.save(f"{MINI_GYM_ROOT_DIR}/scripts/auto_train.py", policy="now")
        # wandb.save(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/legged_robot.py", policy="now")
        # wandb.save(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/legged_robot_config.py", policy="now")
        # wandb.save(f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/__init__.py", policy="now")
        
        # wandb.save(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/__init__.py", policy="now")
        # wandb.save(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/arm_ac.py", policy="now")
        # wandb.save(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/dog_ac.py", policy="now")
        # wandb.save(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/ppo_ac.py", policy="now")
        # wandb.save(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/rollout_storage.py", policy="now")
        

    env = VelocityTrackingEasyEnv(sim_device=args.sim_device, headless=args.headless, cfg=Cfg)
    env = HistoryWrapper(env)
    gpu_id = args.sim_device.split(":")[-1]
    runner = Runner(env, device=f"cuda:{gpu_id}", run_name=args.run_name, resume=args.resume, log_dir=args.log_dir, debug=args.debug)
    runner.learn(num_learning_iterations=args.num_learning_iterations, init_at_random_ep_len=True, eval_freq=args.eval_freq)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Go1")
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--sim_device', type=str, default="cuda:0")
    parser.add_argument('--num_learning_iterations', type=int, default=100000)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--num_envs', type=int, default=2048)
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--tags', nargs='+', default=[])
    parser.add_argument('--notes', type=str, default=None)
    parser.add_argument('--seed', type=int, default=-1)

    args = parser.parse_args()

    if args.debug:
        mode = "disabled"
        args.num_envs = 4
    else:
        mode = "online"
    
        if args.offline:
            mode = "offline"

    if args.no_wandb:
        mode = "disabled"

    if args.resume:
        args.tags.append("resume")

    args.seed = set_seed(args.seed)
    args.tags.append(f"_seed{args.seed}")
    # to see the environment rendering, set headless=False
    train_go1(args)
