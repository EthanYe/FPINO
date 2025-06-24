import numpy as np

size=500
randomIndex=np.random.permutation(size)
datasets=np.load("data/FPINO_dataset.npy")[randomIndex]
# datasets=np.load("data/FPINO_viscoelastic_dataset.npy")[randomIndex]

# decouple and desample
datah=datasets[:,:,::2,0]
dataq=datasets[:,:,::2,1]
reaches = 26
boundaryPoint=18 # input sensor location
length = 500
t1,steps=10,200
x_space = np.linspace(0, length, reaches) 
t_space = np.linspace(0, t1, steps) 
x_all,t_all = np.meshgrid(x_space, t_space)

# Original dataset: x,t,p
dataset=np.zeros((size,200,26,4))
for i in range(size):
   dataset[i,:,:,0]=x_all
   dataset[i,:,:,1]=t_all
   dataset[i,:,:,2]=datah[i,:,:]
   dataset[i,:,:,3]=dataq[i,:,:]  

# Input dataset
inputData=np.zeros((size,200,26,202))
inputData[:,:,:,:2]=dataset[:,:,:,:2]
for k in range(200):
    for i in range(26):
        inputData[:,k,i,2:]=dataset[:,:,boundaryPoint,2]
obsDataHQ=dataset[:,:,:,2:]

# Save data
# prefix="data/FPINO_viscoelastic_"
prefix="data/FPINO_"
np.save(prefix+"inputData.npy",inputData)
np.save(prefix+"full_obsDataHQ.npy",obsDataHQ)