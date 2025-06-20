#CNN初级课
import matplotlib.pyplot as plt
import torch

from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import torch.nn.functional as F 
import torch.optim as optim

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

batch_size=64
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset=datasets.MNIST(root='D:\code\dp\Review-DP\Softmax\mnist',
                             train=True,
                             download=False,
                             transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='D:\code\dp\Review-DP\Softmax\mnist',
                            train=False,
                            download=False,
                            transform=transform)

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=batch_size,
                         )


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)
        # self.pooling =torch.nn.MaxUnpool2d(2)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320,10)

    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.pooling(self.conv1(x)))
        x=F.relu(self.pooling(self.conv2(x)))
        x=x.view(batch_size,-1)
        x=self.fc(x)
        return x
    
model = Net()
    
device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0.0
    for batch_index,(inputs,labels) in enumerate(train_loader,0):
        inputs,labels =inputs.to(device),labels.to(device)
        y_pred=model(inputs)
        loss=criterion(y_pred,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_size % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 300))

def test():
    correct=0
    total=0

    with torch.no_grad():
        for(images,labels) in test_loader:
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _,pred = torch.max(outputs.data,dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('accuracy on test set: %d %%' % (100 * correct / total))
    return correct / total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

