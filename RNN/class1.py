import torch

if __name__=='__main__':
    batch_szie=1
    seq_len=3
    input_size=4
    hidden_size=2

    cell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

    dataset=torch.randn(seq_len,batch_szie,input_size)
    print(dataset)
    hidden=torch.zeros(batch_szie,hidden_size)
    print(hidden)
    for idx,input in enumerate(dataset):
        print('='*20,idx,'='*20)
        print('input_size',input.shape)

        hidden=cell(input,hidden)

        print('outputs size:',hidden.shape)
        print(hidden)
