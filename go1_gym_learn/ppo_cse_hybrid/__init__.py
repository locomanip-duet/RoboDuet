# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import copy
import os
import os.path as osp
import shutil
import statistics
import time
from collections import deque

import cv2
import imageio
import numpy as np
import torch
from ml_logger import logger
from params_proto import PrefixProto

import wandb
from go1_gym import MINI_GYM_ROOT_DIR

from .plan_model import ActorCritic
from .rollout_storage import RolloutStorage
from .ppo import PPO
# from go1_gym.envs.hybrid import HistoryWrapper


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go1_gym_learn.ppo.metrics_caches import DistCache, SlotCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = "/home/pi7113t/dog/dwb-wtw/runs/约束hip_newwidth/2023-12-08/train_ppo/145259.218560"  # updated from load_run and chkpt

class PlannerArgs(PrefixProto, cli=False):
    # planner
    hidden_size = 128
    num_commands = 2
    num_splits = 120
    recurrent = False
    action_type = 'discrete'
    resume_path = ""

def custom_decay_reward_scale(iteration, initial_scale=1.5, final_scale=0.8, max_iterations=8000):
    if iteration >= max_iterations:
        return final_scale
    x = (iteration / max_iterations) ** 2  # Using square to achieve the desired curve
    reward_scale = final_scale + (initial_scale - final_scale) * (1 - x)
    return reward_scale

def custom_increase_reward_scale(iteration, initial_scale=0.2, final_scale=0.7, max_iterations=8000):
    if iteration >= max_iterations:
        return final_scale
    x = (iteration / max_iterations) ** 2  # Using square to achieve the desired curve
    reward_scale = initial_scale + (final_scale - initial_scale) * x
    return reward_scale

class Runner:

    def __init__(self, env, device='cpu', run_name: str=None, resume=False, log_dir=None, debug=False):

        self.device = device
        # self.env: HistoryWrapper = env
        self.env = env
        self.run_name = run_name
        self.log_dir = log_dir
        self.debug = debug
        self.recurrent = PlannerArgs.recurrent
        # if self.log_dir is not None:
        #     self.artifact = wandb.Artifact("checkpoints", type="model")

        actor_critic = ActorCritic(
            num_obs=self.env.num_obs,
            num_history_obs=self.env.cfg.env.num_history_obs,
            hidden_size=PlannerArgs.hidden_size,
            num_commands=PlannerArgs.num_commands,
            num_splits=PlannerArgs.num_splits,
            recurrent=PlannerArgs.recurrent,
            action_type=PlannerArgs.action_type,
            c_action_limits=self.env.cfg.plan.command_limits,
        ).to(self.device)
        

        if resume:
            # load pretrained weights from resume_path
            weights = torch.load(osp.join(PlannerArgs.resume_path, "checkpoints/ac_weights_last.pt"))
            actor_critic.load_state_dict(state_dict=weights)
            print("successfully loaded weights and curriculum !!!")

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, self.env.cfg.env.num_history_obs,
                              num_commands=PlannerArgs.num_commands, c_actions_shape=PlannerArgs.num_splits, hxs_size=PlannerArgs.hidden_size)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, eval_expert=False, width=80, pad=35):

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if hasattr(self.env, "curriculum"):
            caches.__init__(curriculum_bins=len(self.env.curriculum))
        
        hxs = torch.zeros((self.env.num_envs, PlannerArgs.hidden_size), dtype=torch.float, device=self.device)
        masks = torch.ones((self.env.num_envs, 1), device=self.device)
        
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if self.recurrent:
                        origin_actions, hxs, post_actions = self.alg.act(obs[:num_train_envs], hxs[:num_train_envs], masks[:num_train_envs])
                    else:
                        origin_actions, hxs, post_actions = self.alg.act(obs_history[:num_train_envs], hxs[:num_train_envs], masks[:num_train_envs])
                    # if eval_expert:
                    #     actions_eval = self.alg.actor_critic.act_teacher(obs[num_train_envs:],
                    #                                                      privileged_obs[num_train_envs:])
                    # else:
                    #     actions_eval = self.alg.actor_critic.act_student(obs[num_train_envs:],
                    #                                                      obs_history[num_train_envs:])
                    ret = self.env.step(post_actions)
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.env.clear_cached(env_ids)

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    masks = 1 - dones.float()
                    
                    if self.recurrent:
                        self.alg.process_env_step(obs, hxs, rewards, dones, masks, infos)  # TODO: turn to history
                    else:
                        self.alg.process_env_step(obs_history, hxs, rewards, dones, masks, infos)  # TODO: turn to history
                    
                    if self.log_dir is not None:
                        if 'train/episode' in infos:
                            ep_infos.append(infos['train/episode'])

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                stop = time.time()
                collection_time = stop - start
                
                # Learning step
                start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], hxs[:num_train_envs], masks[:num_train_envs])

            # self.env.reward_scales["tracking_lin_vel"] = custom_decay_reward_scale(it, initial_scale=1.5 * self.env.dt, final_scale=1. * self.env.dt)
            # self.env.reward_scales["tracking_ang_vel"] = custom_decay_reward_scale(it, initial_scale=1. * self.env.dt, final_scale=0.8 * self.env.dt)
            # self.env.reward_scales["manip_commands_tracking"] = custom_increase_reward_scale(it, initial_scale=0.2 * self.env.dt, final_scale=0.7 * self.env.dt)
            beta = min(it / 5000, 1)
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start


            if self.log_dir is not None:
                ep_string = f''
                wandb_dict = {}
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
                self.tot_time += learn_time + collection_time
                iteration_time = learn_time + collection_time
                fps = self.num_steps_per_env * self.env.num_envs / iteration_time
                
                for key in ep_infos[0].keys():
                    mean = []
                    for ep_info in ep_infos:
                        mean.append(ep_info[key])
                    mean = torch.mean(torch.stack(mean))
                    
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {mean:.4f}\n"""
                    
                    if not self.debug:
                        wandb_dict['Train_Reward_episode/' + key] = mean
                
                if not self.debug:
                    # wandb_dict["Train_Loss/adaptation_loss"] = mean_adaptation_module_loss
                    wandb_dict["Train_Loss/mean_value_loss"] = mean_value_loss
                    wandb_dict["Train_Loss/mean_surrogate_loss"] = mean_surrogate_loss
            
                    if len(rewbuffer) > 0:
                        wandb_dict['Train_Total_Reward/mean_reward'] = statistics.mean(rewbuffer)
                        wandb_dict['Train_Total_Reward/mean_episode_length'] = statistics.mean(lenbuffer)
                    
                    wandb.log(wandb_dict, step=it)
                str = f" \033[1m Learning iteration {it}/{tot_iter} \033[0m "
        
                log_string = (f"""{'#' * width}\n"""
                    f"""{str.center(width, ' ')}\n\n""")
                log_string += ep_string
                log_string += f"""{'-' * width}\n"""
                log_string += f"""\033[1m{'run_name:':>{pad}} {self.run_name}\033[0m \n"""
                if len(rewbuffer) > 0:
                    log_string += (f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                                    f"""{'Value function loss:':>{pad}} {mean_value_loss:.8f}\n"""
                                    f"""{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.8f}\n"""
                                    # f"""{'Adaptation loss:':>{pad}} {mean_adaptation_module_loss:.8f}\n"""
                                    f"""{'Mean reward (total):':>{pad}} {statistics.mean(rewbuffer):.4f}\n"""
                                    f"""{'Mean episode length:':>{pad}} {statistics.mean(lenbuffer):.4f}\n""")
                                    #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                                    #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
                else:
                    log_string = (f"""{'#' * width}\n"""
                                f"""{str.center(width, ' ')}\n\n"""
                                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                                f"""{'Value function loss:':>{pad}} {mean_value_loss:.8f}\n"""
                                f"""{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.8f}\n""")
                                # f"""{'Adaptation loss:':>{pad}} {mean_adaptation_module_loss:.4f}\n""")

    
                curr_it = it - copy.copy(self.current_learning_iteration)
                eta = self.tot_time / (curr_it + 1) * (num_learning_iterations - curr_it)  # 单位时间乘上剩余次数
                mins = eta // 60
                secs = eta % 60
                log_string += (f"""{'-' * width}\n"""
                            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                            f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
                print(log_string)
                
            if not self.debug and RunnerArgs.save_video_interval:
                self.log_video(it)

            if not self.debug and it % RunnerArgs.save_interval == 0:
                self.save(it)
            ep_infos.clear()

        self.save(it)

    def save(self, it):
        torch.save(self.alg.actor_critic.state_dict(), osp.join(self.log_dir, f"checkpoints/ac_weights_{it:06d}.pt"))
        shutil.copyfile(osp.join(self.log_dir, f"checkpoints/ac_weights_{it:06d}.pt"), 
                        osp.join(self.log_dir, f"checkpoints/ac_weights_last.pt"))


        path = f'{MINI_GYM_ROOT_DIR}/tmp/deploy_model'

        adaptation_module_path = f'{path}/adaptation_module_latest.jit'
        adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
        traced_script_adaptation_module = torch.jit.script(adaptation_module)
        traced_script_adaptation_module.save(adaptation_module_path)

        body_path = f'{path}/body_latest.jit'
        body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
        traced_script_body_module = torch.jit.script(body_model)
        traced_script_body_module.save(body_path)
        
        wandb.save(osp.join(self.log_dir, f"checkpoints/ac_weights_last.pt"))
        wandb.save(adaptation_module_path)
        wandb.save(body_path)
        

    def save_cv(self, frames, it):
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定编码器（此处使用MP4V编码器）
        fourcc = cv2.VideoWriter_fourcc(*'X264')  # 指定编码器（此处使用MP4V编码器）    
        out = cv2.VideoWriter(osp.join(self.log_dir, f'videos/{it:06d}.mp4'), fourcc, int(1 / self.env.dt), (self.env.camera_props.width, self.env.camera_props.height))  # 输出视频文件名、编码器、帧率、分辨率
        for frame in frames:
            out.write(frame[..., :3])
        out.release()
        
    def save_io(self, frames, it):
        # 设置视频编码器和帧速率
        writer = imageio.get_writer(osp.join(self.log_dir, f'videos/{it:06d}.mp4'), fps=int(1 / self.env.dt))
        for frame in frames:
            writer.append_data(frame[..., :3])
        writer.close()
            
    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            
            self.save_io(frames, it)
            
            frames = np.stack(frames, axis=0)
            # Channels should be (time, channel, height, width) or (batch, time, channel, height width)
            frames = np.transpose(frames, (0, 3, 1, 2))
            # wandb.log({"video": wandb.Video(frames, fps=1 / self.env.dt, format='mp4')})
            # wandb.run.summary["latest_video"] = wandb.Video(frames, fps=1 / self.env.dt, format='mp4')
            

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                # wandb.log({"video": wandb.Video(frames, fps=1 / self.env.dt, format='mp4')})
                # wandb.run.summary["latest_video"] = wandb.Video(frames, fps=1 / self.env.dt, format='mp4')

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
