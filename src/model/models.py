import torch
from torch import nn


class ConvNext(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int = 9):
        super().__init__()
        self.model = model
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)
        layer = model.features[0][0]
        new_in_channels = 4
        new_layer = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
        )
        copy_weights = 0
        new_layers_weight = new_layer.weight.clone()
        new_layers_weight[:, : layer.in_channels, :, :] = layer.weight.clone()
        for i in range(new_in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layers_weight[:, channel: channel + 1, :, :] = layer.weight[
                :, copy_weights: copy_weights + 1, ::
            ].clone()
        new_layer.weight = nn.Parameter(new_layers_weight)
        self.model.features[0][0] = new_layer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        return features


class ResNet(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int = 9) -> None:
        super().__init__()
        self.model = model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        layer = self.model.conv1
        new_in_channels = 4
        new_layer = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
        )
        copy_weights = 0
        new_layers_weight = new_layer.weight.clone()
        new_layers_weight[:, : layer.in_channels, :, :] = layer.weight.clone()
        for i in range(new_in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layers_weight[:, channel: channel + 1, :, :] = layer.weight[
                :, copy_weights: copy_weights + 1, ::
            ].clone()
        new_layer.weight = nn.Parameter(new_layers_weight)
        self.model.conv1 = new_layer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        return features


class Dummy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 1, (7, 7), (2, 2), (3, 3))
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 64))
        self.linear = nn.Linear(64, 9)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv1(images)
        features = self.pool(features)
        features = self.linear(features)
        return torch.squeeze(features)
