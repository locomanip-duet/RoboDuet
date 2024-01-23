
class GlobalSwitch():
    def __init__(self) -> None:
        self.swith_flag = False


    def open_switch(self):
        self.swith_flag = True
    
    @property    
    def swith_open(self):
        return self.swith_flag
    
    
global_switch = GlobalSwitch()