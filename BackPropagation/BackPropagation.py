import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=torch.tensor([1.0])
w.requires_grad=True

def forward(x):
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y-y_pred)**2

print("predict(before training)",4,forward(4).item())

epochs=[]
costs=[]

for epoch in range(100):
    epochs.append(epoch)

    for x,y in zip(x_data,y_data):
        l = loss(x,y)
        l.backward()
        print(f"{x},{y},{round(w.grad.item(),2)}")
        w.data -= 0.01 *w.grad.item()

        w.grad.data.zero_()
    
    costs.append(l.item())
    print(f"process is {epoch, l.item()}")

print(f"predict is (after training) is {4,round(forward(4).item()),2}")

plt.plot(epochs,costs)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()