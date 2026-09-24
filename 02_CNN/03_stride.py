import torch
import torch.nn as nn

image = torch.randn(1,3,32,32)

conv1 = nn.Conv2d(
    in_channels = 3,
    out_channels = 16,
    kernel_size = 3,
    padding = 1,
    stride = 1
)

conv2 = nn.Conv2d(
    in_channels = 16,
    out_channels = 32,
    kernel_size = 3,
    padding = 1,
    stride = 1
)

output1 = conv1(image)
output2 = conv2(output1)

print(image.shape)
print(output2.shape)

# when using stride is 2 , shape is 
# torch.Size([1, 32, 8, 8])

# when using stride is 1, shape
# torch.Size([1, 32, 32, 32])

# increasing stride always decreases the size of feature map

