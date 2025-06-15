import numpy as np
import matplotlib.pyplot as plt

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=1.0
w_list=[]
loss_list=[]
grad_list=[]
epouchs=[]
def forward(x):
    return x*w

def cost(xs,ys):
    cost=0
    for x,y in zip(xs,ys):
        y_pred=forward(x)
        cost += (y_pred-y)**2
    
    return cost/len(ys)

def gradient(xs,ys):
    grad=0
    for x,y in zip(xs,ys):
        grad += 2*x*(x*w-y)
    
    return grad/len(xs)

print("predict (before training)",4,forward(4))

for epouch in range(100):
    epouchs.append(epouch)
    cost_val = cost(x_data,y_data)
    loss_list.append(cost_val)
    grad_val = gradient(x_data,y_data)
    grad_list.append(grad_val)
    w -= 0.01 * grad_val
    w_list.append(w)
    print(f"epouch is {epouch},w is {w}, loss is {cost_val}")

print("predict (after training)",4,forward(4))

plt.plot(epouchs,loss_list)
plt.ylabel("loss")
plt.xlabel("epouch")
plt.show()