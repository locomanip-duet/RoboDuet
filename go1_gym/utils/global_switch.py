import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GlobalSwitch():
    def __init__(self) -> None:
        self.switch_flag = False
        self.count = 0
        self.hybrid_reward_scales = None
        self.pretrained_reward_scales = None
        
        self.pretrained_to_hybrid_start = 20
        self.pretrained_to_hybrid_end = self.pretrained_to_hybrid_start + 20
        
    
    def init_sigmoid_lr(self):
        range_len = self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start
        divide = np.linspace(-7, 7, range_len)
        self.lr_down = 1-sigmoid(divide)
        
    def init_linear_lr(self):
        range_len = self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start
        self.lr_down = np.linspace(1, 0, range_len)
        
    def set_reward_scales(self, hybrid_reward_scales, pretrained_reward_scales):
        self.hybrid_reward_scales = hybrid_reward_scales
        self.pretrained_reward_scales = pretrained_reward_scales
        
    def get_reward_scales(self):
        if self.count < self.pretrained_to_hybrid_start:
            return self.pretrained_reward_scales
        
        elif self.count < self.pretrained_to_hybrid_end:
            reward_scales = {}
            lr = self.lr_down[self.count - self.pretrained_to_hybrid_start]
            for key, end in self.hybrid_reward_scales.items():
                start = self.pretrained_reward_scales[key]
                # reward_scales[key] = start + (end - start) * (self.count - self.pretrained_to_hybrid_start) / (self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start)
                reward_scales[key] = start * lr + end * (1 - lr)
            
            return reward_scales
        
        else:
            return self.hybrid_reward_scales
    
    # def get_reward_scales(self):
    #     if self.count < self.pretrained_to_hybrid_start:
    #         return self.pretrained_reward_scales
        
    #     elif self.count < self.pretrained_to_hybrid_end:
    #         reward_scales = {}
    #         for key, end in self.hybrid_reward_scales.items():
    #             start = self.pretrained_reward_scales[key]
    #             reward_scales[key] = start + (end - start) * (self.count - self.pretrained_to_hybrid_start) / (self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start)
    #         return reward_scales
        
    #     else:
    #         return self.hybrid_reward_scales
    
    def get_beta(self):
        if self.count <= self.pretrained_to_hybrid_start:
            return 0.0
        
        elif self.count < self.pretrained_to_hybrid_end:
            return 0.5 * (self.count - self.pretrained_to_hybrid_start) / (self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start)
        
        else:
            return 0.5
    
    def open_switch(self):
        self.switch_flag = True
    
    @property    
    def switch_open(self):
        return self.switch_flag
    
    
global_switch = GlobalSwitch()