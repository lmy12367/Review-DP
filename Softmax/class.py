import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

batch_size=64
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307),(0.3081))

    ]
)
train_dataset = datasets.MNIST(root="D:\code\dp\Review-DP\Softmax\mnist",
                               train= True,download=False,transform= transform)

train_loader = DataLoader(train_dataset,shuffle= True,batch_size=batch_size)

test_dataset = datasets.MNIST(root="D:\code\dp\Review-DP\Softmax\mnist",
                              download=False,transform=transform)
test_loader = DataLoader(test_dataset,shuffle = True,batch_size = batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1=torch.nn.Linear(784,512)
        self.linear2=torch.nn.Linear(512,256)
        self.linear3=torch.nn.Linear(256,128)
        self.linear4=torch.nn.Linear(128,64)
        self.linear5=torch.nn.Linear(64,10)

    def forward(self,x):
        x=x.view(-1,784)
        x=F.relu(self.linear1(x))
        x=F.relu(self.linear2(x))
        x=F.relu(self.linear3(x))
        x=F.relu(self.linear4(x))
        return self.linear5(x)
    

model=Net()

critertion=torch.nn.CrossEntropyLoss()
op = torch.optim.SGD(model.parameters(),lr=0.01)

epochs=[]
costs=[]
correctloss=[]

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        op.zero_grad()

        outputs = model(inputs)
        loss = critertion(outputs,target)

        loss.backward()
        op.step()
        running_loss += loss.item()
        costs.append(running_loss)

        if batch_idx %300 == 299:
            print('[%d,%5d] loss: %3f' % (epoch+1,batch_idx+1,running_loss / 300))
            running_loss = 0.0

def vali():
    corrrect=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs = model(images)
            _,predict=torch.max(outputs.data,dim=1)

            total += labels.size(0)
            corrrect += (predict == labels).sum().item()
        print('Accuracy on test set: %d %%'%(100 * corrrect/ total))
        correctloss.append(corrrect)


if __name__ == '__main__':
    for epoch in range(10):
        epochs.append(epoch)
        train(epoch)
        vali()


        

    