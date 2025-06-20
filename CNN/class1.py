#卷积操作
import torch

in_channel, out_channel = 5,10
width,height=100,100
kernel_size=3
batch_size=1

input = torch.randn(batch_size,
                    in_channel,
                    width,
                    height)

conv_layer= torch.nn.Conv2d(in_channel,
                            out_channel,
                            kernel_size=kernel_size)

outputs=conv_layer(input)

print(input.shape)
print(outputs)
print(conv_layer.weight.shape)