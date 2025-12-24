import numpy as np

class Train:
    def __init__(self,model):
        self.model = model
        self.reverse_model = np.flip(model)

    def Fit(self,batchs,labels,lr,epochs):
        self.batchs = batchs
        self.labels = labels
        self.lr = lr
        for epoch in range(epochs+1):
            self.all_loss = 0
            for i in range(len(self.batchs)):
                for layer in range(len(self.model)):
                    if layer == 0:
                        self.summary(self.batchs[i],self.model[layer])
                        self.relu(self.model[layer])
                    else:
                        if layer != len(self.model)-1:
                            self.summary(self.model[layer-1].output,self.model[layer])
                            self.relu(self.model[layer])
                        else:
                            self.summary(self.model[layer-1].output,self.model[layer])
                self.calc_loss(self.labels[i])
                self.grad(self.labels[i])
                self.w_update(self.batchs[i])
                self.spicker(epoch,self.labels[i])
            # self.log(self.labels,self.batchs,self.all_loss,epoch)

    # no activate 1000batch 0.001 100ep layer2 327sec loss2.33
    # activate 1000batch 0.001 100ep layer2  340sec lossnan

    def summary(self,input,layer):
            layer.output = layer.w_summ(input)
    
    def calc_loss(self,label):
        self.all_loss += (self.model[-1].output - label)**2

    def spicker(self,epoch,label):
        print(f"Epoch: {epoch+1} \n Predict: {self.model[-1].output} Label: {label}")

    def grad(self,label):
        for i in range(len(self.reverse_model)):
            if i == 0:
                self.reverse_model[i].delta = self.model[-1].output - label
            else:
                self.reverse_model[i].delta = np.dot(self.reverse_model[i-1].delta,self.reverse_model[i-1].w.T)*self.relu2deriv(self.reverse_model[i].output)
    
    def logger(self):
        print(f"Inputs: {self.inputs.shape} \n Labels: {self.labels.shape}")
        for i in self.model:
            print(f"W: {i.w.shape} \n Output: {i.output.shape} \n Delta: {i.delta.shape}")

    def w_update(self,input):
        for i in range(len(self.model)):
            if i == 0:
                self.model[i].w -= self.lr * np.outer(input,self.model[i].delta)
            else:
                self.model[i].w -= self.lr * np.outer(self.model[i-1].output,self.model[i].delta)

    def relu(self,layer):
        layer.act_output = [max(0,i) for i in layer.output]
    
    def relu2deriv(self,output):
        return [min(1,i) for i in output]