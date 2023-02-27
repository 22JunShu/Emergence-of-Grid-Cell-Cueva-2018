import torch
from gridcell_model import RNNEC
from walk import data_gen

neuron_num = 100
input_dim = 2
output_dim = 2
length_range = 5
time_step = 500
trial_num = 500
initial_xu =(torch.full((trial_num,neuron_num),0,dtype=torch.float),torch.full((trial_num,neuron_num),0,dtype=torch.float))

# 生成速度和位置数据
data = data_gen(trial_num,time_step,length_range)
input_data = torch.stack(data[1:],dim = 0)
input_data = torch.squeeze(input_data).permute(1,2,0)
y_real = data[0]
u_zeros = torch.zeros((trial_num,time_step,neuron_num))
myECModel = RNNEC(input_dim,neuron_num,output_dim)

# 学习参数设置
lr_init = 0.005
optim_wdecay = torch.optim.Adam(myECModel.parameters(), lr=lr_init, weight_decay=1e-2) # weight decay 控制权重带来的损失
loss_fn = torch.nn.MSELoss() 

iter_time = 200
for epoch in range(iter_time):
    initial_xu =(torch.full((trial_num,neuron_num),0,dtype=torch.float),torch.full((trial_num,neuron_num),0,dtype=torch.float))
    output_x,output_u,output_y = myECModel(input_data,initial_xu)
    loss_total = loss_fn(output_y,y_real) + loss_fn(output_u,u_zeros) #位置损失+代谢损失
    print(loss_total,epoch)

    optim_wdecay.zero_grad()

    loss_total.backward()

    optim_wdecay.step()
