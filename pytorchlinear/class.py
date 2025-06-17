import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


x_data = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0],[5.0]])



class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()


criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

epochs = []
costs = []

for epoch in range(100):
    epochs.append(epoch)
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    costs.append(loss.item())
    print(epoch,loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print('w = ',model.linear.weight.item())
print('b = ',model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ',y_test.data)


plt.plot(epochs,costs)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()
