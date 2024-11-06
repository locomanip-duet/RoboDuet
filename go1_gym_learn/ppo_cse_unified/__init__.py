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
from params_proto import PrefixProto

import wandb
from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.automatic import HistoryWrapper
from go1_gym.utils import global_switch

from .unified2head_ac import Unified2ActorCritic
from .ppo import PPO


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

class UnifiedRunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 400
    log_freq = 10
    log_video = True

    # load and resume
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = "your_unified_ckpt_path"  # updated from load_run and chkpt
    resume = False

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
        self.env: HistoryWrapper = env
        self.run_name = run_name
        self.log_dir = log_dir
        self.debug = debug
        self.num_steps_per_env = UnifiedRunnerArgs.num_steps_per_env
        
        self.unified_model = Unified2ActorCritic(
            self.env.cfg.env.num_observations,
            self.env.cfg.env.num_privileged_obs,
            self.env.cfg.env.num_obs_history,
            self.env.cfg.env.num_actions,
            device=self.device,
        ).to(self.device)


        if UnifiedRunnerArgs.resume:
            # load pretrained weights from resume_path
            weights = torch.load(UnifiedRunnerArgs.resume_path)
            self.unified_model.load_state_dict(state_dict=weights)
            print("successfully loaded weights!!!")


        self.alg = PPO(self.unified_model, device=self.device)
        self.alg.init_storage(
            self.env.num_train_envs,
            self.num_steps_per_env, 
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_obs_history],
            [self.env.num_actions])


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

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        ep_infos = []
        beta = 0
        
        tot_iter = self.current_learning_iteration + num_learning_iterations
        fake_actions_arm = torch.zeros(self.env.num_envs, self.env.num_actions_arm, dtype=torch.float, device=self.device, requires_grad=False)
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])

                    actions_dog = actions_train[..., :self.env.cfg.dog.dog_actions]
                    if global_switch.switch_open:                    
                        actions_arm = actions_train[..., self.env.cfg.dog.dog_actions:]
                    else:
                        actions_arm = fake_actions_arm
                        
                    ret = self.env.step(actions_dog, actions_arm)
                    
                    rewards_dog, rewards_arm, dones, infos = ret
                    
                    obs_dict = self.env.get_observations()
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                    "obs_history"]
                    
                    obs, privileged_obs, obs_history, rewards_dog, rewards_arm, dones = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device), rewards_dog.to(self.device), rewards_arm.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(torch.stack([rewards_dog[:num_train_envs], rewards_arm[:num_train_envs]], dim=-1), dones[:num_train_envs], infos)

                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.env.clear_cached(env_ids)

                    if self.log_dir is not None:
                        if 'train/episode' in infos:
                            ep_infos.append(infos['train/episode'])

                        cur_reward_sum += rewards_dog
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
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

            mean_value_loss_dog, mean_value_loss_arm, mean_surrogate_loss_dog, mean_surrogate_loss_arm, mean_adaptation_module_loss, mean_adaptation_module_test_loss = self.alg.update(beta)
            stop = time.time()
            learn_time = stop - start
            
            global_switch.count += 1

            if it == global_switch.pretrained_to_hybrid_start:
                blue_bold_text = '\033[1;34m'
                reset_color = '\033[0m'
                print(blue_bold_text + '=' * 160 + '\n'
                      + 'Unified Policy Output: Start to gradually integrate arm and leg rewards.' + '\n'
                      + '=' * 160 + reset_color )
                
                global_switch.open_switch()
                change_setting = vars(self.env.cfg.hybrid.rewards)
                for key, value in change_setting.items():
                    setattr(self.env.cfg.rewards, key, value)
                        
            beta = global_switch.get_beta()
            
            if self.log_dir is not None:
                ep_string = f''
                wandb_dict = {}
                wandb_dict["Efficiency/collect_time"] = collection_time
                wandb_dict["Efficiency/learn_time"] = learn_time
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
                
                dog_action_std = self.alg.actor_critic.std_dog.clone()
                arm_action_std = self.alg.actor_critic.std_arm.clone()
                if not self.debug:
                    wandb_dict["Train_Loss/mean_value_loss_arm"] = mean_value_loss_arm
                    wandb_dict["Train_Loss/mean_surrogate_loss_arm"] = mean_surrogate_loss_arm
                    wandb_dict["Train_Loss/mean_value_loss_dog"] = mean_value_loss_dog
                    wandb_dict["Train_Loss/mean_surrogate_loss_dog"] = mean_surrogate_loss_dog
                    wandb_dict["Train_std/arm_action_std"] = arm_action_std.mean()
                    wandb_dict["Train_std/dog_action_std"] = dog_action_std.mean()      
            
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
                                    f"""{'Arm action std:':>{pad}} {arm_action_std.cpu().tolist()}\n"""
                                    f"""{'Dog action std:':>{pad}} {dog_action_std.cpu().tolist()}\n"""
                                    f"""{'Arm Value function loss:':>{pad}} {mean_value_loss_arm:.8f}\n"""
                                    f"""{'Arm Surrogate loss:':>{pad}} {mean_surrogate_loss_arm:.8f}\n"""
                                    f"""{'Dog Value function loss:':>{pad}} {mean_value_loss_dog:.8f}\n"""
                                    f"""{'Dog Surrogate loss:':>{pad}} {mean_surrogate_loss_dog:.8f}\n"""
                                    f"""{'Test Adaptation loss:':>{pad}} {mean_adaptation_module_test_loss:.8f}\n"""
                                    f"""{'Train Adaptation loss:':>{pad}} {mean_adaptation_module_loss:.8f}\n"""
                                    f"""{'Mean reward (total):':>{pad}} {statistics.mean(rewbuffer):.4f}\n"""
                                    f"""{'Mean episode length:':>{pad}} {statistics.mean(lenbuffer):.4f}\n""")
                    
                else:
                    log_string = (f"""{'#' * width}\n"""
                                f"""{str.center(width, ' ')}\n\n"""
                                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                                f"""{'Arm Value function loss:':>{pad}} {mean_value_loss_arm:.8f}\n"""
                                f"""{'Arm Surrogate loss:':>{pad}} {mean_surrogate_loss_arm:.8f}\n"""
                                f"""{'Dog Value function loss:':>{pad}} {mean_value_loss_dog:.8f}\n"""
                                f"""{'Dog Surrogate loss:':>{pad}} {mean_surrogate_loss_dog:.8f}\n"""
                                f"""{'Test Adaptation loss:':>{pad}} {mean_adaptation_module_test_loss:.8f}\n"""
                                f"""{'Train Adaptation loss:':>{pad}} {mean_adaptation_module_loss:.8f}\n""")
    
                curr_it = it - copy.copy(self.current_learning_iteration)
                eta = self.tot_time / (curr_it + 1) * (num_learning_iterations - curr_it)
                mins = eta // 60
                secs = eta % 60
                log_string += (f"""{'-' * width}\n"""
                            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                            f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
                print(log_string)
                
                with open(osp.join(self.log_dir, "log.txt"), "a") as f:
                    f.write(log_string)
                
            if UnifiedRunnerArgs.save_video_interval and UnifiedRunnerArgs.log_video:
                self.log_video(it)

            if not self.debug and it % UnifiedRunnerArgs.save_interval == 0:
                self.save(it)
                
            ep_infos.clear()

        self.save(it)

    def save(self, it):
        torch.save(self.alg.actor_critic.state_dict(), osp.join(self.log_dir, f"checkpoints_unified/ac_weights_{it:06d}.pt"))
        shutil.copyfile(osp.join(self.log_dir, f"checkpoints_unified/ac_weights_{it:06d}.pt"),
            osp.join(self.log_dir, f"checkpoints_unified/ac_weights_last.pt"))
        
        if it in [44800, 44400, 44000]:
            torch.save(self.alg.actor_critic.state_dict(), osp.join(self.log_dir, f"checkpoints_unified/ac_weights_{it:06d}.pt"))
            wandb.save(osp.join(self.log_dir, f"checkpoints_unified/ac_weights_{it:06d}.pt"))

        path = osp.join(self.log_dir, f"deploy_model")
        adaptation_module_path = f'{path}/adaptation_module_latest.jit'
        adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
        traced_script_adaptation_module_dog = torch.jit.script(adaptation_module)
        traced_script_adaptation_module_dog.save(adaptation_module_path)
        body_path = f'{path}/body_latest.jit'
        body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
        traced_script_body_module = torch.jit.script(body_model)
        traced_script_body_module.save(body_path)

        # save to wandb
        wandb.save(adaptation_module_path)
        wandb.save(body_path)
        wandb.save(osp.join(self.log_dir, f"checkpoints_unified/ac_weights_last.pt"))

    def save_cv(self, frames, it):
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'X264')    
        out = cv2.VideoWriter(osp.join(self.log_dir, f'videos/{it:06d}.mp4'), fourcc, int(1 / self.env.dt), (self.env.camera_props.width, self.env.camera_props.height))
        for frame in frames:
            out.write(frame[..., :3])
        out.release()
        
    def save_io(self, frames, it):
        writer = imageio.get_writer(osp.join(self.log_dir, f'videos/{it:06d}.mp4'), fps=int(1 / self.env.dt))
        for frame in frames:
            writer.append_data(frame[..., :3])
        writer.close()
            
    def log_video(self, it):
        if it - self.last_recording_it >= UnifiedRunnerArgs.save_video_interval:
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
