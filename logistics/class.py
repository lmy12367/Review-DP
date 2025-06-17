import numpy as np
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[0.0],[0.0],[1.0]])
 
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)
op=torch.optim.SGD(model.parameters(),lr=0.01)

epochs=[]
costs=[]

for epoch in range(100):
    epochs.append(epoch)
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    costs.append(loss.item())
    print(epoch,loss.item())

    op.zero_grad()
    loss.backward()
    op.step()

print(f"w = {model.linear.weight.item()}")
print(f"b =  {model.linear.bias.item()}")

X=np.linspace(0,10,200)
x_test = torch.Tensor(X).view(200,1)
y_test = model(x_test)
Y = y_test.data.numpy()
plt.ylabel('Probability of Pass')
plt.xlabel('Hours')
plt.plot(X, Y)
plt.grid()
plt.show()

plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.plot(epochs, costs)
plt.show()


