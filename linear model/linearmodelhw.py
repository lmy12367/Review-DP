# 线性模型作业
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x_data=[1.0,2.0,3.,4.0]
y_data=[3.0,6.0,9.0,12.0]
w_list=np.arange(0.0,5.0,0.1)
b_list=np.arange(-2.0,2.0,0.1)

mse_list=[]
ww,bb = np.meshgrid(w_list,b_list)

def forward(x):
    return x*w+b

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)

for b in b_list:
    for w in w_list:
        print(f"w is {w},b is{b}")
        loss_num=0
        for x_val,y_val in zip(x_data,y_data):
            y_pred_val=forward(x_val)
            loss_val=loss(x_val,y_val)
            loss_num += loss_val
            print(f"x is {x_val},y is{y_val},lossval is {loss_val}")
        
        print(f"mse is {loss_num/3}")
        mse_list.append(loss_num/3)


mse = np.array(mse_list).reshape(w_list.shape[0],b_list.shape[0])


   