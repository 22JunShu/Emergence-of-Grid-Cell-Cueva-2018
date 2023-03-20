import torch
from walk import data_gen

neuron_num = 20
input_dim = 2
output_dim = 2
length_range = 5
time_step = 1000000
test_num = 2
#batch_size = 500

myECModel = torch.load("myECModel_18_1.pth")


# hyperparameters
# lr_init = 0.00001
# optim_wdecay = torch.optim.LBFGS(myECModel.parameters(), lr=lr_init, max_iter=20) 
loss_fn = torch.nn.MSELoss(reduction="mean") 


num_epoch = 10
metabolic_ratio = 0.1
penalty_ratio = 1
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

    # loss_total_test = loss_fn(output_y_test,y_real_test) + metabolic_ratio*loss_fn(output_u_test,u_zeros_test)+penalty_ratio*(l2_penalty(W_in) +l2_penalty(W_out))
    # loss_pos_test = loss_fn(output_y_test,y_real_test)
    # print("test loss:",loss_total_test.item(),"test position loss:",loss_pos_test.item())
    # test_loss.append(loss_total_test.item())

# output_u_test = [trial_num,time_step,neurons]
# output_y_test = [trial_num,time_step,position]
from matplotlib import pyplot as plt
# plt.scatter(y_list[0],y_list[1],c=u_list[19],cmap = "coolwarm")
# plt.show()
import numpy as np
import seaborn as sns
plt.figure(figsize=[8,6])
for neuron in range(neuron_num):      
    u_avg_over_trials = np.zeros((20,20))
    print("neuron",neuron+1)
    for j in range(test_num):
        u_list = output_u_test[j].numpy().T #u_list = [neurons,time_step]
        y_list = y_real_test[j].numpy().T #y_list = [position,time_step]
        u_total = np.zeros((20,20))
        time_total = np.zeros((20,20))

        for i in range(time_step):
            x = int((y_list[0][i]+5)*2)
            y = int((y_list[1][i]+5)*2)
            u_total[x,y] += u_list[neuron][i]
            time_total[x,y] += 1
        u_avg = u_total/time_total
        u_avg_over_trials += u_avg
    u_avg_over_trials = u_avg_over_trials/test_num
    plt.subplot(4,5,neuron+1)
    plt.axis('off')
    #sns.heatmap(u_avg,vmin = -1,vmax = 1,cmap = 'viridis')
    sns.heatmap(u_avg,cmap = 'viridis')
plt.show()
#plt.show()
# j=0
# print(input_data_test.size())
# u_list = output_u_test[j].numpy().T #u_list = [neurons,time_step]
# theta_list = input_data_test[j].numpy().T[1]
# print(theta_list)
# speed_list = input_data_test[j].numpy().T[0]
# for neuron in range(neuron_num):      
#     print("neuron",neuron+1) 
#     plt.subplot(10,10,neuron+1)
#     plt.scatter(speed_list,u_list[neuron],s = 0.1)

# plt.show()

# trace_sim = output_y_test[0].numpy().T
# trace_real = y_real_test[0].numpy().T
# import numpy as np
# loss_p1 = np.mean((trace_sim[0] - trace_real[0])**2)
# loss_p2 = np.mean((trace_sim[1] - trace_real[1])**2)
# print("x_pos_loss:",loss_p1,"y_pos_loss:",loss_p2)
# print("W_out_1_avg:",torch.var(W_out[0]).item(),"W_out2_avg:",torch.var(W_out[1]).item())
# # plt.figure()
# # plt.subplot(1,2,1)
# # plt.plot(trace_sim[0],trace_sim[1],label = "sim")
# # plt.xlim((-5,5))
# # plt.ylim((-5,5))
# # plt.legend()
# # plt.subplot(1,2,2)
# # plt.plot(trace_real[0],trace_real[1],label = "real")
# # plt.xlim((-5,5))
# # plt.ylim((-5,5))
# # plt.legend()
# # plt.title("x_loss%.4f;y_loss%.4f"%(loss_p1,loss_p2))
# # plt.show()

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

# loss_p1_total = []
# loss_p2_total = []
# fig = plt.figure(figsize=[10,10])
# for i in range(test_num):
#     trace_sim = output_y_test[i].numpy().T
#     trace_real = y_real_test[i].numpy().T
#     if i < 16:
#         ax = fig.add_subplot(4,4,i+1)
#         ax.plot(trace_sim[0],trace_sim[1],label = "sim")
#         ax.plot(trace_real[0],trace_real[1],label = "real")
#         ax.set_xlim((-5,5))
#         ax.set_ylim((-5,5))
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     loss_p1 = np.mean((trace_sim[0] - trace_real[0])**2)
#     loss_p2 = np.mean((trace_sim[1] - trace_real[1])**2)
#     loss_p1_total.append(loss_p1)
#     loss_p2_total.append(loss_p2)
# plt.legend(loc = "right")
# plt.title("myECModel")
# plt.show()
# print("x_pos_loss_avg:%.4f,y_pos_loss_avg:%.4f"%(np.mean(loss_p1_total),np.mean(loss_p2_total)))
# print("x_pos_loss_std:%.4f,y_pos_loss_std:%.4f"%(np.std(loss_p1_total),np.std(loss_p2_total)))
