import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len =xy.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len


# dataset = DiabetesDataset('D:\\code\\dp\\Review-DP\\DataLoad\\diabetes.csv.gz')
# train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid =torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
    


# model=Model()
# criterion = torch.nn.BCELoss(reduction='mean')
# op = torch.optim.SGD(model.parameters(),lr=0.01)
# epochs=[]
# costs=[]
if __name__ == '__main__':
    dataset = DiabetesDataset('D:\\code\\dp\\Review-DP\\DataLoad\\diabetes.csv.gz')
    train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
    # train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)
    model=Model()
    criterion = torch.nn.BCELoss(reduction='mean')
    op = torch.optim.SGD(model.parameters(),lr=0.01)
    epochs=[]
    costs=[]
    for epoch in range(100):
        epochs.append(epoch)
        loss_num=0
        for i, data in enumerate(train_loader,0):
            inputs,labers =data
            y_pred = model(inputs)
            loss = criterion(y_pred,labers)
            loss_num += loss.item()
            print(epoch,i,loss.item())

            op.zero_grad()
            loss.backward()
            op.step()
        
        costs.append(loss_num/(i+1))

    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(epochs, costs)
    plt.show()