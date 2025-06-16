import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import _cm
from mpl_toolkits.mplot3d import Axes3D

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w_1 = torch.Tensor([1.0])
w_2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])
w_1.requires_grad = True
w_2.requires_grad = True
b.requires_grad = True


def forward(x):
    return x **2 * w_1 + x *w_2 + b

def loss(x,y):
    y_pred=forward(x)
    return (y-y_pred)**2

print("predict(before training)",4,round(forward(4).item(),2))
epochs=[]
costs=[]

for epoch in range(100):
        epochs.append(epoch)

        for x,y in zip(x_data,y_data):
            l = loss(x,y)
            l.backward()
            print('\tgrad：', x, y, round(w_1.grad.item(), 2),round(w_2.grad.item(), 2))
            w_1.data -= 0.01 *w_1.grad.item()
            w_2.data -= 0.01 *w_2.grad.item()
            b.data -= 0.01 *b.grad.item()

            w_1.grad.data.zero_()
            w_2.grad.data.zero_()
            b.grad.data.zero_()
        
        costs.append(l.item())
        print(f"progress{epoch,l.item()}")

print('Predict (after training)',4,round(forward(4).item(),2))

plt.plot(epochs,costs)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()     
