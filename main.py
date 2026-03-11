import data_coder
from neuron import Node
import analitycs_utils as au

data_dir = r"D:\GitHub\Neural-network\src\datasets\unzip_datasets\mnist"

allTime = 256 #ms
step_time = 1 #ms

timeLines, labelsList = data_coder.timing_coder(data_dir,allTime=allTime,count=50,power=1)

node1 = Node((784,10))

for epoch in range(1000):
    print(epoch)
    for i, timeline in enumerate(timeLines):
        node1.step(timeline,labelsList[i])
    #au.plot_all_weights(node1.W, epoch)
au.plot_all_weights(node1.W, 1)