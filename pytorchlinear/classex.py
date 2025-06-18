import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0]])



class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model1 = LinearModel()  
model2 = LinearModel()
model3 = LinearModel()
model4 = LinearModel()
model5 = LinearModel()
#model6 = LinearModel()
model7 = LinearModel()
model8 = LinearModel()
models = [model1,model2,model3,model4,model5,model7,model8]

criterion = torch.nn.MSELoss(size_average = False)

op1 = torch.optim.SGD(model1.parameters(),lr = 0.01)
op2 = torch.optim.Adagrad(model2.parameters(),lr = 0.01)
op3 = torch.optim.Adam(model3.parameters(),lr = 0.01)
op4 = torch.optim.Adamax(model4.parameters(),lr = 0.01)
op5 = torch.optim.ASGD(model5.parameters(),lr = 0.01)
#op6 = torch.optim.LBFGS(model6.parameters(),lr = 0.01)
op7 = torch.optim.RMSprop(model7.parameters(),lr = 0.01)
op8 = torch.optim.Rprop(model8.parameters(),lr = 0.01)
ops = [op1,op2,op3,op4,op5,op7,op8]

titles = ['SGD','Adagrad','Adam','Adamax','ASGD','RNSprop','Rprop']

fig,ax = plt.subplots(2,4,figsize = (20,10),dpi = 100)

index = 0
for item in zip(ops,models):
    epochs = []
    costs = []

    model = item[1]
    op = item[0]

    for epoch in range(100):
        epochs.append(epoch)
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        costs.append(loss.item())
        print(epoch, loss)

        op.zero_grad()
        loss.backward()
        op.step()

    print('w = ', model.linear.weight.item())
    print('b = ', model.linear.bias.item())

    x_test = torch.Tensor([[4.0]])
    y_test = model(x_test)
    print('y_pred = ', y_test.data)

    a = int(index /4)
    b = int(index % 4)
    ax[a][b].set_ylabel('Cost')
    ax[a][b].set_xlabel('Epoch')
    ax[a][b].set_title(titles[index])
    ax[a][b].plot(epochs, costs)
    index += 1
plt.show()
