import torch
import torch.nn as nn

image = torch.randn(1,3,32,32)

# conv = nn.Conv2d(
#     in_channels = 3,
#     out_channels = 16,
#     kernel_size = 3,
#     padding = 0
# )

# with padding 
conv = nn.Conv2d(
    in_channels = 3,
    out_channels = 16,
    kernel_size = 3,
    padding = 1
)


output = conv(image)

print("input shape ", image.shape)
print("output shape ", output.shape)