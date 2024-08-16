import torch
import time

logdir = "/home/elgceben/sim_go1/runs/funny/lpyrpy/"

print("################################ for wtw #########################")
body = torch.jit.load(logdir+'checkpoints_wtw/body_latest.jit')
adaptation_module = torch.jit.load(logdir+'checkpoints_wtw/adaptation_module_latest.jit')
obs1 = torch.zeros(2102).to('cpu')
obs2 = torch.zeros(2100).to('cpu')
begin = time.time()
for item in range(1000):
    body.forward(obs1)
    adaptation_module.forward(obs2)
print("infer_time for wtw: ", time.time() - begin)


print("################################ for wtw #########################")
body = torch.jit.load(logdir+'checkpoints_wtw/body_latest.jit')
adaptation_module = torch.jit.load(logdir+'checkpoints_wtw/adaptation_module_latest.jit')
obs1 = torch.zeros(2102).to('cpu')
obs2 = torch.zeros(2100).to('cpu')
begin = time.time()
for item in range(100):
    body.forward(obs1)
    adaptation_module.forward(obs2)
print("infer_time for wtw: ", time.time() - begin)

print("################################ for dog #########################")
body = torch.jit.load(logdir+'checkpoints_dog/body_dog_latest.jit')
adaptation_module = torch.jit.load(logdir+'checkpoints_dog/adaptation_module_dog_latest.jit')
obs1 = torch.zeros(1442).to('cpu')
obs2 = torch.zeros(1440).to('cpu')
begin = time.time()
for item in range(100):
    body.forward(obs1)
    adaptation_module.forward(obs2)
print("infer_time for dog: ", time.time() - begin)

print("################################ for arm #########################")
body = torch.jit.load(logdir+'checkpoints/body_latest.jit')
adaptation_module = torch.jit.load(logdir+'checkpoints/adaptation_module_latest.jit')
obs1 = torch.zeros(639).to('cpu')
obs2 = torch.zeros(630).to('cpu')
begin = time.time()
for item in range(100):
    body.forward(obs1)
    adaptation_module.forward(obs2)
print("infer_time for arm: ", time.time() - begin)
