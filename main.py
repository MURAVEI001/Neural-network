import data_coder
from neuron import Node
import analitycs_utils as au

data_dir = r"D:\GitHub\Neural-network\src\datasets\unzip_datasets\mnist"

allTime = 256 #ms
step_time = 1 #ms

imagesTrain, labelsTrain, imagesValid, labelsValid = data_coder.getTimeLine(data_dir,Train=100,Valid=10,allTime=allTime,power=1)

node1 = Node((784,10))

for epoch in range(100):
    print(epoch+1)
    for i, timeline in enumerate(imagesTrain):
        node1.Fit(timeline,labelsTrain[i])
for i, timeline in enumerate(imagesValid):
    node1.Valid(timeline,labelsValid[i])
au.plot_all_weights(node1.W, 1)