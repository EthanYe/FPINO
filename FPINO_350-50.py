import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from torch.utils.data import Dataset
from torchsummary import summary
class PINODataset(Dataset):
    def __init__(self, x_train,t_train,p_train,h_train):
        self.x_train = x_train.astype(np.float32)
        self.t_train = t_train.astype(np.float32)
        self.p_train = p_train.astype(np.float32)
        self.h_train = h_train.astype(np.float32)


    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return {
            'x_train': torch.tensor(self.x_train[idx], requires_grad=True),
            't_train': torch.tensor(self.t_train[idx], requires_grad=True),
            'p_train': torch.tensor(self.p_train[idx], requires_grad=True),
            'h_train': torch.tensor(self.h_train[idx], requires_grad=True)
        }

class PINODataset_obs(Dataset):
    def __init__(self, x_train,t_train,p_train,x_obs,t_obs,p_obs,h_obs):
        self.x_train = x_train.astype(np.float32)
        self.t_train = t_train.astype(np.float32)
        self.p_train = p_train.astype(np.float32)
        self.x_obs = x_obs.astype(np.float32)
        self.t_obs = t_obs.astype(np.float32)
        self.p_obs = p_obs.astype(np.float32)
        self.h_obs = h_obs.astype(np.float32)


    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return {
            'x_train': torch.tensor(self.x_train[idx], requires_grad=True),
            't_train': torch.tensor(self.t_train[idx], requires_grad=True),
            'p_train': torch.tensor(self.p_train[idx], requires_grad=True),
            'x_obs': torch.tensor(self.x_obs[idx], requires_grad=True),
            't_obs': torch.tensor(self.t_obs[idx], requires_grad=True),
            'p_obs': torch.tensor(self.p_obs[idx], requires_grad=True),
            'h_obs': torch.tensor(self.h_obs[idx], requires_grad=True)
        }

class PINO(nn.Module):
    def __init__(self,lb,ub,branchLayers,trunkLayers):
        super(PINO, self).__init__()
        self.lb,self.ub=torch.from_numpy(lb.astype(np.float32)).to(device),torch.from_numpy(ub.astype(np.float32)).to(device)
        self.k = torch.tensor(10, device=device)
        self.w_f = torch.tensor(1, device=device)
        self.w_sim = torch.tensor(1, device=device)
        self.w_real = torch.tensor(1, device=device)
        self.g = torch.tensor(9.81, device=device)
        self.D = torch.tensor(0.2, device=device)
        self.A = torch.tensor(3.14 * (0.2 ** 2) / 4, device=device)
        self.fric = torch.tensor(0.02, device=device)
        self.max_q = torch.tensor(0.1, device=device)
        self.max_h = torch.tensor(100, device=device)
        self.a = torch.tensor(1000, device=device)
        self.a1 = torch.tensor(0.1, device=device)
        self.a2 = torch.tensor(200, device=device)
        self.num_outputs=2
        self.sigmoid = nn.Sigmoid()  # Instantiate the Sigmoid function
        self.t_obs=torch.linspace(0,10,200, device=device)
        self.t_obs1=self.t_obs.unsqueeze(0)
        self.branchLayers=nn.ModuleList()
        for i in range(len(branchLayers)-1):
            self.branchLayers.append(nn.Linear(branchLayers[i],branchLayers[i+1]))
        self.trunkLayers=nn.ModuleList()
        for i in range(len(trunkLayers)-1):
            self.trunkLayers.append(nn.Linear(trunkLayers[i],trunkLayers[i+1]))
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)])

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = torch.einsum("bi,bi->b", x_func, x_loc)
        y = torch.unsqueeze(y, dim=1)
        y += self.b[index]
        return y
    def forward(self,x,t,p):
        
        xt = torch.cat([x,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        xt = 2.0 * (xt[:,:2]  - self.lb[:2] ) / (self.ub[:2]  - self.lb[:2] ) - 1.0
        p = 2.0 * (p  - self.lb[2] ) / (self.ub[2]  - self.lb[2] ) - 1.0
        # pi=p[:]
        # information filter
        pi = p * self.sigmoid(1-self.k * (self.t_obs1 - t))
        for layer in self.branchLayers[:-1]:
            pi= torch.tanh(layer(pi))
        pi= self.branchLayers[-1](pi)
        xti=xt[:]
        for layer in self.trunkLayers[:-1]:
            xti= torch.tanh(layer(xti))
        xti= self.trunkLayers[-1](xti)

        # Split x_func into respective outputs
        shift = 0
        size = xti.shape[1]
        xs = []
        for i in range(self.num_outputs):
            pi_ =pi[:, shift : shift + size]
            x = self.merge_branch_trunk(pi_, xti, i)
            xs.append(x)
            shift += size
        u=  torch.concat(xs, dim=1)
        u[:,0:1],u[:,1:]=u[:,0:1]*self.max_h ,u[:,1:]*self.max_q 
        return u
    
    ## PDE as loss function. Thus would use the network which we call as u_theta
    def f(self,x,t,p):
        
        u = self.forward(x,t,p)
        h,q=u[:,0:1] ,u[:,1:]
        h_x = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
        h_t = torch.autograd.grad(h.sum(), t, create_graph=True)[0]
        q_x = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
        q_t = torch.autograd.grad(q.sum(), t, create_graph=True)[0]
        ff = (self.A * q_t + q * q_x + self.g * self.A * self.A * h_x+self.fric*torch.abs(q)*q/2/self.D)/self.a1
        gf = (self.A * h_t + q * h_x + self.a * self.a * q_x / self.g)/self.a2
        return ff,gf
    
    def trainNetwork(self,train_loader,train_loader2):
        iterations = 100000
        optimizer = torch.optim.Adam(self.parameters())
        mse_cost_function = torch.nn.MSELoss() # Mean squared error
        directory="models/"+uname+"/"
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        for epoch in range(iterations):
            for batch,batch2 in zip(train_loader,train_loader2):
                x_train = batch['x_train'].to(device).reshape(-1,1)
                t_train = batch['t_train'].to(device).reshape(-1,1)
                p_train = batch['p_train'].to(device).reshape(-1,200)
                h_train = batch['h_train'].to(device).reshape(-1,1)

                x_train2 = batch2['x_train'].to(device).reshape(-1,1)
                t_train2 = batch2['t_train'].to(device).reshape(-1,1)
                p_train2 = batch2['p_train'].to(device).reshape(-1,200)
                # hq_train2 = batch2['hq_train'].to(device).reshape(-1,2)
                x_obs2=batch2['x_obs'].to(device).reshape(-1,1)
                t_obs2=batch2['t_obs'].to(device).reshape(-1,1)
                p_obs2= batch2['p_obs'].to(device).reshape(-1,200)
                h_obs2=batch2['h_obs'].to(device).reshape(-1,1)
                optimizer.zero_grad() # to make the gradients zero
                #LOSS
                # Loss based on boundary conditions
                
                net_bc_out= self.forward(x_train, t_train,p_train) # output of u(x,t)
                mse_uh = mse_cost_function(net_bc_out[:,0:1], h_train)/self.max_h/self.max_h
               
                f_out,g_out = self.f(x_train, t_train,p_train) # output of f(x,t)
                all_zeros = torch.zeros_like(f_out)
                mse_f = mse_cost_function(f_out, all_zeros)+mse_cost_function(g_out, all_zeros)

                net_bc_out2= self.forward(x_obs2, t_obs2,p_obs2) # output of u(x,t)
                mse_uh2 = mse_cost_function(net_bc_out2[:,0:1], h_obs2)/self.max_h/self.max_h
                f_out2,g_out2 = self.f(x_train2, t_train2,p_train2) # output of f(x,t)
                all_zeros = torch.zeros_like(f_out2)
                mse_f2 = mse_cost_function(f_out2, all_zeros)+mse_cost_function(g_out2, all_zeros)

                # Combining the loss functions
                loss = self.w_sim * mse_uh +   + self.w_f*(mse_f2+mse_f)+self.w_real*mse_uh2
                loss.backward() # This is for computing gradients using backward propagation
                optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Training Loss: {loss.item()}")
            if epoch % 1000 == 0:
                # Save Model
                torch.save(model.state_dict(), directory+uname+ f"w_f{self.w_f}_model-"+str(epoch)+".pt")

length=500
t1=10
condtionSize=500
trainingSize1=350
trainingSize2=400-trainingSize1
batchSize=15
# original data
randomIndex=np.random.permutation(condtionSize)
print(randomIndex)
'''load dataset 1'''
prefix="data/FPINO"
inputData=np.load(prefix+'_inputData.npy')[randomIndex]# [200,200,26,202]

obsDataH=np.load(prefix+"_full_obsDataHQ.npy")[randomIndex,:,:,0] # [300,200,26,1]
print("Data loaded.")
# seperate training/testing data 
trainInput=inputData[:trainingSize1]# [200,200,26,202]
testInput=inputData[trainingSize1:]# [100,200,26,202]

# training input (collocation points)
x_train=trainInput[:,:,:,0:1]
t_train=trainInput[:,:,:,1:2]
p_train=trainInput[:,:,:,2:]
h_train=obsDataH[:trainingSize1] #
train_dataset = PINODataset(x_train,t_train,p_train, h_train)
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)  # Assuming `train_dataset` is already defined


'''load dataset 2'''

prefix2="data/FPINO_viscoelastic"
obsPoints =[18,24,14]
inputData2=np.load(prefix2+'_inputData.npy')[randomIndex]# [200,200,26,202]
obsDataH2=np.load(prefix2+"_full_obsDataHQ.npy")[randomIndex,:,:,0] # [300,200,26,1]
print("Data loaded.")
# seperate training/testing data 
trainInput2=inputData2[:trainingSize2]# [200,200,26,202]
testInput2=inputData2[trainingSize2:]# [100,200,26,202]

# training input (collocation points)
x_train2=trainInput2[:,:,:,0:1]
t_train2=trainInput2[:,:,:,1:2]
p_train2=trainInput2[:,:,:,2:]
x_obs2=trainInput2[:,:,obsPoints,0:1] # [100,200,3]
t_obs2=trainInput2[:,:,obsPoints,1:2]# [100,200,3]
p_obs2=trainInput2[:,:,obsPoints,2:]# [100,200,3,200]
h_obs2=obsDataH2[:trainingSize2,:,obsPoints] # [100,200,3]
train_dataset2 = PINODataset_obs(x_train2,t_train2,p_train2,x_obs2,t_obs2,p_obs2, h_obs2)
train_loader2 = DataLoader(train_dataset2, batch_size=5, shuffle=True)  # Assuming `train_dataset` is already defined

lb = np.array([0., 0,-200])
ub = np.array([length, t1,200])  # x and t?

lenObs=len(obsPoints) # number
# After data reshaping, right before initializing the model

branchLayers = [200,60,60,60,60,60,60,60,60,60]
trunkLayers = [2, 30,30,30,30,30,30,30,30,30]

model = PINO( lb, ub, branchLayers,trunkLayers).to(device)
info=summary(model)
uname=f"trainingSize-{trainingSize1}-{trainingSize2}"
print(uname)
np.save(prefix+uname+'-randomIndex.npy',randomIndex)
model.trainNetwork(train_loader,train_loader2=train_dataset2)


