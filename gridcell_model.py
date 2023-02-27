import torch
from torch import nn,Tensor
from torch.nn import Module,Parameter
import torch.nn.functional as func
from walk import data_gen
# CTRNN网络的模型
class RNNGridCell(Module):
    def __init__(self,input_size,hidden_size,output_size) -> None:
        super(RNNGridCell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_rec = Parameter(Tensor(hidden_size,hidden_size))
        self.weight_in = Parameter(Tensor(hidden_size,input_size))
        self.weight_out = Parameter(Tensor(output_size,hidden_size))
        self.bias = Parameter(Tensor(hidden_size))
    def forward(self,input,xu):
        x,u = xu
        delta = func.linear(input,self.weight_in,self.bias) + func.linear(u,self.weight_rec) + torch.randn(self.hidden_size)
        x = 0.9*x + 0.1*(delta + torch.randn(self.hidden_size))
        u = torch.tanh(x)
        y = func.linear(u,self.weight_out)
        return x,u,y

class RNNEC(Module):
    def __init__(self,input_size,hidden_size,output_size) -> None:
        super(RNNEC,self).__init__()
        self.input_size = input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.RNNlayer = RNNGridCell(input_size,hidden_size,output_size)
    def forward(self,input,initial_xu):
         #input: trial_num*time_step*2;
         #initial_xu: (trial_num*neurons,trial_num*neurons)
        length = input.size(1) 
        xu = initial_xu 
        output_x = []
        output_u = []
        output_y = []
        for t in range(length):
            i_t = input[:,t,:]
            i_t = torch.squeeze(i_t) #i_t: trial_num*2
            states = self.RNNlayer(i_t,xu)
            xu = states[:2]
            '''print(states[0].size()) # x:trial_num*neurons
            print(states[1].size()) # u:trial_num*neurons
            print(states[2].size()) # y:trial_num*out_put(2)'''
            output_x.append(states[0])
            output_u.append(states[1])
            output_y.append(states[2])
        output_x = torch.stack(output_x) 
        # time_step*trial_num*neurons

        output_u = torch.stack(output_u)
        # time_step*trial_num*neurons

        output_y = torch.stack(output_y)
        # time_step*trial_num*output_size


        output_x=output_x.permute(1,0,2)
        # trial_num*time_step*neurons
        output_y=output_y.permute(1,0,2)

        output_u=output_u.permute(1,0,2)
        return  output_x,output_u,output_y
