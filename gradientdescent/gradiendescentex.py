import numpy as np
import random
import matplotlib.pyplot as plt

x_data=[1.0,2.0,3.0]
y_data=[2.0,3.0,4.0]

w=1.0

def forward(x):
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y-y_pred)**2

def gradient(x,y):
    return 2*x*(x*w-y)

print("predict befor",4,forward(4))

epouchs=[]
costs=[]

for epouch in range(100):
    index = random.randint(0,2)
    x=x_data[index]
    y=y_data[index]
    grad = gradient(x,y)

    w -= 0.01 * grad 

    print(f"{x},{y}",round(grad,2))

    l=loss(x,y)

    epouchs.append(epouch)
    costs.append(l)
    print('Epoch:',epouch,'w=',round(w,2),'loss=',round(l,2))
print('Predict (after training)',4,round(forward(4),2))


plt.plot(epouchs,costs)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()
