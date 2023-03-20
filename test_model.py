import torch
from walk import data_gen

neuron_num = 20
input_dim = 2
output_dim = 2
length_range = 5
time_step = 500
trial_num = 1500
batch_size = 500

myECModel = torch.load("myECModel_18_1.pth")


# hyperparameters
# lr_init = 0.00001
# optim_wdecay = torch.optim.LBFGS(myECModel.parameters(), lr=lr_init, max_iter=20) 
loss_fn = torch.nn.MSELoss(reduction="mean") 


num_epoch = 10
metabolic_ratio = 0.1
penalty_ratio = 1
test_num = 200
# train_dataset = TensorDataset(input_data, y_real)
# train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# train_loss = []
test_loss = []

# initialize x and u
# initial_xu =(torch.full((batch_size,neuron_num),0,dtype=torch.float),torch.full((batch_size,neuron_num),0,dtype=torch.float))
    
def l2_penalty(w):
    return torch.mean(w.pow(2))     
myECModel.eval()
params = myECModel.state_dict()
W_in = params["RNNlayer.weight_in"]
W_out = params["RNNlayer.weight_out"]
with torch.no_grad():
    data_test = data_gen(test_num,time_step,length_range)
    input_data_test = torch.stack(data_test[1:],dim = 0)
    input_data_test = torch.squeeze(input_data_test).permute(1,2,0)
    y_real_test = data_test[0]
    u_zeros_test = torch.zeros((test_num,time_step,neuron_num))

    initial_xu_test =(torch.full((test_num,neuron_num),0,dtype=torch.float),torch.full((test_num,neuron_num),0,dtype=torch.float))
    output_x_test,output_u_test,output_y_test = myECModel(input_data_test,initial_xu_test)

    #loss_total_test = loss_fn(output_y_test,y_real_test) + metabolic_ratio*loss_fn(output_u_test,u_zeros_test)+penalty_ratio*(l2_penalty(W_in) +l2_penalty(W_out))
    loss_u_test = metabolic_ratio*loss_fn(output_u_test,u_zeros_test)
    loss_w_test = penalty_ratio*(l2_penalty(W_in) +l2_penalty(W_out))
    loss_pos_test = loss_fn(output_y_test,y_real_test)
    loss_total_test = loss_fn(output_y_test,y_real_test) + metabolic_ratio*loss_fn(output_u_test,u_zeros_test)+penalty_ratio*(l2_penalty(W_in) +l2_penalty(W_out))
    print("test loss:",loss_total_test.item(),"test position loss:",loss_pos_test.item(),"\ntest metabolic loss:",loss_u_test.item(),"test weight loss:",loss_w_test.item())
    test_loss.append(loss_total_test.item())


from matplotlib import pyplot as plt
# trace_sim = output_y_test[0].numpy().T
# trace_real = y_real_test[0].numpy().T
import numpy as np
# loss_p1 = np.mean((trace_sim[0] - trace_real[0])**2)
# loss_p2 = np.mean((trace_sim[1] - trace_real[1])**2)
# print("x_pos_loss:",loss_p1,"y_pos_loss:",loss_p2)
# print("W_out_1_avg:",torch.var(W_out[0]).item(),"W_out2_avg:",torch.var(W_out[1]).item())
# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(trace_sim[0],trace_sim[1],label = "sim")
# plt.xlim((-5,5))
# plt.ylim((-5,5))
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(trace_real[0],trace_real[1],label = "real")
# plt.xlim((-5,5))
# plt.ylim((-5,5))
# plt.legend()
# plt.title("x_loss%.4f;y_loss%.4f"%(loss_p1,loss_p2))
# plt.show()

# fig = plt.figure(figsize=(5,5))
# ax1 = fig.add_subplot(111)
# ax1.plot(trace_sim[0],trace_sim[1],label = "sim")
# ax1.plot(trace_real[0],trace_real[1],label = "real")
# ax1.set_xlim((-5,5))
# ax1.set_ylim((-5,5))
# ax1.get_yaxis().set_visible(False)
# ax1.get_xaxis().set_visible(False)
# plt.legend()
# plt.show()

loss_p1_total = []
loss_p2_total = []
fig = plt.figure(figsize=[10,10])
plot_num = 0
for i in range(test_num):
    trace_sim = output_y_test[i].numpy().T
    trace_real = y_real_test[i].numpy().T
    loss_p1 = np.mean((trace_sim[0] - trace_real[0])**2)
    loss_p2 = np.mean((trace_sim[1] - trace_real[1])**2)
    loss_p1_total.append(loss_p1)
    loss_p2_total.append(loss_p2)
    if loss_p1 >= 2 and plot_num < 16:
        plot_num += 1
        ax = fig.add_subplot(4,4,plot_num)
        ax.plot(trace_sim[0],trace_sim[1],label = "sim")
        ax.plot(trace_real[0],trace_real[1],label = "real")
        ax.set_xlim((-5,5))
        ax.set_ylim((-5,5))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
plt.legend(loc = "right")
plt.title("myECModel")
plt.show()
print("x_pos_loss_avg:%.4f,y_pos_loss_avg:%.4f"%(np.mean(loss_p1_total),np.mean(loss_p2_total)))
print("x_pos_loss_std:%.4f,y_pos_loss_std:%.4f"%(np.std(loss_p1_total),np.std(loss_p2_total)))
plt.figure(figsize=[9,4])
plt.subplot(1,2,1)
plt.hist(loss_p1_total)
plt.subplot(1,2,2)
plt.hist(loss_p2_total)
plt.show()