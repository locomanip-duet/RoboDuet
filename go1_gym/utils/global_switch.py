
class GlobalSwitch():
    def __init__(self) -> None:
        self.switch_flag = False
        self.count = 0
        self.hybrid_reward_scales = None
        self.pretrained_reward_scales = None
        
        self.pretrained_to_hybrid_start = 10000
        self.pretrained_to_hybrid_end = 12000
        
    def set_reward_scales(self, hybrid_reward_scales, pretrained_reward_scales):
        self.hybrid_reward_scales = hybrid_reward_scales
        self.pretrained_reward_scales = pretrained_reward_scales
        
    def get_reward_scales(self):
        if self.count < self.pretrained_to_hybrid_start:
            return self.pretrained_reward_scales
        
        elif self.count < self.pretrained_to_hybrid_end:
            reward_scales = {}
            for key, end in self.hybrid_reward_scales.items():
                start = self.pretrained_reward_scales[key]
                reward_scales[key] = start + (end - start) * (self.count - self.pretrained_to_hybrid_start) / (self.pretrained_to_hybrid_end - self.pretrained_to_hybrid_start)
            return reward_scales
        
        else:
            return self.hybrid_reward_scales
    
    def open_switch(self):
        self.switch_flag = True
    
    @property    
    def switch_open(self):
        return self.switch_flag
    
    
global_switch = GlobalSwitch()