import torch
from gridcell_model import RNNEC
from walk import data_gen
from torch.utils.data import TensorDataset, DataLoader

neuron_num = 20
lr_init = 0.0001
num_epoch = 200
metabolic_ratio = 0.5

input_dim = 2
output_dim = 2
length_range = 5
time_step = 200
trial_num = 1500
batch_size = 500
initial_xu =(torch.full((batch_size,neuron_num),0,dtype=torch.float),torch.full((batch_size,neuron_num),0,dtype=torch.float))

# generate speed and location
data = data_gen(trial_num,time_step,length_range)
input_data = torch.stack(data[1:],dim = 0)
input_data = torch.squeeze(input_data).permute(1,2,0)
y_real = data[0]
u_zeros = torch.zeros(batch_size,time_step,neuron_num)
#myECModel = RNNEC(input_dim,neuron_num,output_dim)
myECModel = torch.load("myECModel_18.pth")
# hyperparameters

optim_wdecay = torch.optim.Adam(myECModel.parameters(), lr=lr_init, weight_decay=0) # weight decay: a penalty for large parameters
loss_fn = torch.nn.MSELoss() 


test_num = 10
train_dataset = TensorDataset(input_data, y_real)
train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loss = []
test_loss = []

# initialize x and u
initial_xu =(torch.full((batch_size,neuron_num),0,dtype=torch.float),torch.full((batch_size,neuron_num),0,dtype=torch.float))
    
for epoch in range(num_epoch):
    print("epoch:",epoch+1)
    myECModel.train()
    loss_total = 0
    for i, (input_batch,y_batch) in enumerate(train_loader):
        output_x,output_u,output_y = myECModel(input_batch,initial_xu)
        
        loss_batch = loss_fn(output_y,y_batch) + metabolic_ratio*loss_fn(output_u,u_zeros) #位置损失+代谢损失
        
        loss_pos = loss_fn(output_y,y_batch)
        optim_wdecay.zero_grad()
        loss_batch.backward()
        optim_wdecay.step()

        loss_total += loss_batch.item()
    loss_total /= len(train_loader)
    print("train loss:",loss_total,"train position_loss:",loss_pos.item())
    train_loss.append(loss_total)

    myECModel.eval()
    with torch.no_grad():
        data_test = data_gen(test_num,time_step,length_range)
        input_data_test = torch.stack(data_test[1:],dim = 0)
        input_data_test = torch.squeeze(input_data_test).permute(1,2,0)
        y_real_test = data_test[0]
        u_zeros_test = torch.zeros((test_num,time_step,neuron_num))

        initial_xu_test =(torch.full((test_num,neuron_num),0,dtype=torch.float),torch.full((test_num,neuron_num),0,dtype=torch.float))
        output_x_test,output_u_test,output_y_test = myECModel(input_data_test,initial_xu_test)

        loss_total_test = loss_fn(output_y_test,y_real_test) + metabolic_ratio*loss_fn(output_u_test,u_zeros_test)
        loss_pos_test = loss_fn(output_y_test,y_real_test)
        print("test loss:",loss_total_test.item(),"test position loss:",loss_pos_test.item())
        test_loss.append(loss_total_test.item())


torch.save(myECModel,"myECModel_18_1.pth")
import matplotlib.pyplot as plt
trace_sim = output_y_test[0].numpy().T
trace_real = y_real_test[0].numpy().T
# plt.figure(figsize=[11,3])
# plt.subplot(1,3,1)
# plt.plot(trace_sim[0],trace_sim[1],label = "sim")
# plt.xlim((-5,5))
# plt.ylim((-5,5))
# plt.legend()
# plt.subplot(1,3,2)
# plt.plot(trace_real[0],trace_real[1],label = "real")
# plt.xlim((-5,5))
# plt.ylim((-5,5))
# plt.legend()
fig = plt.figure(figsize=(11,5))
ax1 = fig.add_subplot(121)
ax1.plot(trace_sim[0],trace_sim[1],label = "sim")
ax1.plot(trace_real[0],trace_real[1],label = "real")
ax1.set_xlim((-5,5))
ax1.set_ylim((-5,5))
plt.legend()
ax2 = fig.add_subplot(1,2,2)
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()