from torch import randn
import torch
from torch import tensor
# square space
# lrange = 5 length of side = 10

def oneStep(step_direction,y,lrange):
    step_length = abs(randn(1))*0.1
    step_direction_new = step_direction + randn(1)/5
    y_new = y + torch.tensor([step_length*torch.cos(step_direction_new*torch.pi),step_length*torch.sin(step_direction_new*torch.pi)],dtype=torch.float)
    if abs(y_new[0])< lrange and abs(y_new[1])< lrange:
        return y_new,step_direction_new,step_length
    else: #out of range
        return oneStep(step_direction,y,lrange) # resample
def trial(step,lrange): #step: Number of time steps in one trial
    #position =tensor([0,0])# initial position
    position = torch.rand((2))*lrange 
    position_list = []
    speed_list = []
    direction_list = []
    step_direction = tensor(0)
    for t in range(step):
        position,step_direction,step_length = oneStep(step_direction,position,lrange)
        position_list.append(position)
        speed_list.append(step_length)
        direction_list.append(step_direction)
    position_list = torch.stack(position_list)
    speed_list = torch.stack(speed_list)
    direction_list = torch.stack(direction_list)
    return position_list,speed_list,direction_list

def data_gen(trial_num,step,lrange):
    position = []
    speed = []
    direction = []
    for i in range(trial_num):
        position_list,speed_list,direction_list = trial(step,lrange)
        position.append(position_list)
        speed.append(speed_list)
        direction.append(direction_list)
    position = torch.stack(position) # trial_num*time_step*2
    speed = torch.stack(speed) # trial_num*time_step*1
    direction = torch.stack(direction)# trial_num*time_step*1
    return position,speed,direction #位置，速度大小，速度方向
'''
trial_num = 2
step = 50
input = torch.stack(data_gen(trial_num,step)[1:],dim = 0)
input = torch.squeeze(input).permute(1,2,0)
print(input.size())
print(input)
y = data_gen(trial_num,step)[0][0]

import matplotlib.pyplot as plt
y = y.numpy().T
print(y)
plt.plot(y[0,:],y[1,:])
plt.show()
'''

