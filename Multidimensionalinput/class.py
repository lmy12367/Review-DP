import numpy as np
import torch
import matplotlib.pyplot as plt

xy=np.loadtxt('D:\\code\\dp\\Review-DP\\Multidimensionalinput\\diabetes.csv.gz',delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])      #最后一列不要 ，向量形式
y_data = torch.from_numpy(xy[:,[-1]])       #只要最后一列，且为矩阵形式


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,2)
        self.linear4 = torch.nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):

        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x
    

model= Model()

criterion = torch.nn.BCELoss(reduction='mean')
op = torch.optim.SGD(model.parameters(),lr=0.1)

epochs=[]
costs=[]

for epoch in range(10000):
    epochs.append(epoch)

    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)

    costs.append(loss.item())
    print(f'{epoch,loss.item()}')

    op.zero_grad()
    loss.backward()

    op.step()

plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.plot(epochs, costs)
plt.show() 