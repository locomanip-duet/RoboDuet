
class GlobalSwitch():
    def __init__(self) -> None:
        self.switch_flag = False


    def open_switch(self):
        self.switch_flag = True
    
    @property    
    def switch_open(self):
        return self.switch_flag
    
    
global_switch = GlobalSwitch()