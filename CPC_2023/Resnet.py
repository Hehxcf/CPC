import torchvision
from torch import nn


class Resnet50(nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base_net = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        feature = self.base_net(x)
        feature = feature.view(-1, 2048)
        out = self.fc(feature)
        softmax_out = nn.Softmax(dim=1)(out)
        return feature, out, softmax_out


class Resnet101(nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        resnet50 = torchvision.models.resnet101(pretrained=True)
        self.base_net = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        feature = self.base_net(x)
        feature = feature.view(-1, 2048)
        out = self.fc(feature)
        softmax_out = nn.Softmax(dim=1)(out)
        return feature, out, softmax_out


class Resnet18(nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        resnet50 = torchvision.models.resnet18(pretrained=True)
        self.base_net = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feature = self.base_net(x)
        feature = feature.view(-1, 512)
        out = self.fc(feature)
        softmax_out = nn.Softmax(dim=1)(out)
        return feature, out, softmax_out
