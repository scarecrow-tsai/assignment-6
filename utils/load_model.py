import torch
from torch._C import Value
import torch.nn as nn


def count_model_params(model, is_trainable=True):
    if is_trainable:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())

    return num_params


class ModelCL(nn.Module):
    def __init__(self, num_classes, norm_type):
        super(ModelCL, self).__init__()

        self.num_classes = num_classes
        self.norm_type = norm_type

        # block 1
        self.block1 = nn.Sequential(
            self.conv_block(
                c_in=1,
                c_out=8,
                norm_type=self.norm_type,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            self.conv_block(
                c_in=8,
                c_out=12,
                norm_type=self.norm_type,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            self.conv_block(
                c_in=12,
                c_out=8,
                norm_type=self.norm_type,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        # block 2
        self.block2 = nn.Sequential(
            self.conv_block(
                c_in=8,
                c_out=12,
                norm_type=self.norm_type,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            self.conv_block(
                c_in=12,
                c_out=16,
                norm_type=self.norm_type,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            self.conv_block(
                c_in=16,
                c_out=12,
                norm_type=self.norm_type,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        # block 3
        self.block3 = nn.Sequential(
            self.conv_block(
                c_in=12,
                c_out=16,
                norm_type=self.norm_type,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self.conv_block(
                c_in=16,
                c_out=16,
                norm_type=self.norm_type,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.final_conv = nn.Conv2d(
            in_channels=16,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.final_conv(x)
        x = x.squeeze()

        return x

    def conv_block(self, c_in, c_out, norm_type, **kwargs):
        if norm_type == "bnorm":
            seq_block = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
                nn.BatchNorm2d(num_features=c_out),
                nn.ReLU(),
            )
        elif norm_type == "gnorm":
            seq_block = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
                nn.GroupNorm(num_groups=c_out // 2, num_channels=c_out),
                nn.ReLU(),
            )
        elif norm_type == "lnorm":
            seq_block = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
                nn.GroupNorm(num_groups=1, num_channels=c_out),
                nn.ReLU(),
            )
        else:
            raise ValueError("Incorrect norm_type.")

        return seq_block


if __name__ == "__main__":

    # classification models
    print("Classification Model (num params):")
    model = ModelCL(num_classes=10, norm_type="lnorm")
    inp = torch.rand(2, 1, 28, 28)
    print(f"Input: {inp.shape}")
    print(f"Output: {model(inp).shape}")
    print(f"Num parameters: {count_model_params(model)}")
