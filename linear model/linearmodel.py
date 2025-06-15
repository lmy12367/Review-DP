# 线性模型
import numpy as np
import matplotlib.pyplot as plt

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

def forward(x):
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)

W_list=[]
mse_list=[]

for w in np.arange(0.0,4.4,0.1):
    print("w:",w)
    l_sum=0
    for x_val,y_val in zip(x_data,y_data):
        y_pred_val=forward(x_val)
        loss_val=loss(x_val,y_val)
        l_sum += loss_val
        print(f"{x_val,y_val,y_pred_val,loss_val}")
    print(f"MSE is {l_sum/3}")
    W_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(W_list,mse_list)
plt.ylabel("loss")
plt.xlabel("w")
plt.show()