import torch
import torch.nn as nn

image = torch.randn(1,3,32,32)

conv = nn.Conv2d(
    in_channels = 3,
    out_channels = 16,
    kernel_size = 3,
    padding = 1,
    stride = 1
)

# max pooling
# pool = nn.MaxPool2d(
#     kernel_size = 2,
#     stride = 2
# )

pool = nn.AdaptiveAvgPool2d((1,1))

output1 = conv(image)
output2 = pool(output1)

print(image.shape)
print(output2.shape)