import data_coder
from neuron import Node
import analitycs_utils as au
import torch

data_dir = r"D:\GitHub\Neural-network\src\datasets\unzip_datasets\mnist"

allTime = 256 #ms
step_time = 1 #ms

torch.set_grad_enabled(False)

device = torch.device("cuda")

imagesTrain, labelsTrain, imagesValid, labelsValid = data_coder.getTimeLine(data_dir,Train=2,Valid=1,
                                                                            allTime=allTime,power=1,device=device)

node1 = Node((784,10),device=device)

for epoch in range(1):
    print(epoch+1)
    node1.Fit(imagesTrain,labelsTrain)
au.plot_all_weights(node1.W, 1)