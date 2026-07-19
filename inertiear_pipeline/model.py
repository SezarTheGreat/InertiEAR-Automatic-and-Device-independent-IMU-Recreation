import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, dropout=0.3):
        super(DenseLayer, self).__init__()
        # Bottleneck design
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        # x is a list of tensors or a concatenated tensor
        if isinstance(x, list):
            x = torch.cat(x, 1)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.dropout(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, dropout=0.3, memory_efficient=False):
        super(DenseBlock, self).__init__()
        self.memory_efficient = memory_efficient
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, dropout)
            self.layers.append(layer)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.layers:
            if self.memory_efficient and init_features.requires_grad:
                def closure(*args, l=layer):
                    concatenated_features = torch.cat(args, 1)
                    return l(concatenated_features)
                new_features = torch.utils.checkpoint.checkpoint(closure, *features, use_reentrant=False)
            else:
                new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class InertiEAR_DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, dropout=0.3, num_classes=7, memory_efficient=False):
        super(InertiEAR_DenseNet, self).__init__()
        
        # Initial convolution (1 channel input -> grayscale spectrograms)
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks & Transitions
        in_channels = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=in_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout=dropout,
                memory_efficient=memory_efficient
            )
            self.features.add_module(f'denseblock{i+1}', block)
            in_channels = in_channels + num_layers * growth_rate
            
            # Transition layer if not the last block
            if i != len(block_config) - 1:
                trans = Transition(in_channels=in_channels, out_channels=in_channels // 2)
                self.features.add_module(f'transition{i+1}', trans)
                in_channels = in_channels // 2
                
        # Final Batch Normalization
        self.features.add_module('norm5', nn.BatchNorm2d(in_channels))
        
        # Classification layer
        self.classifier = nn.Linear(in_channels, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def get_piecewise_momentum_optimizer(model, base_lr=0.1, momentum=0.9, weight_decay=1e-4):
    """
    Returns an SGD optimizer with momentum.
    Piecewise learning rate decay is handled using PyTorch's MultiStepLR scheduler.
    """
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer

def get_lr_scheduler(optimizer, milestones=[30, 60, 80], gamma=0.1):
    """
    Returns a MultiStepLR scheduler that decays the learning rate at specified epoch milestones.
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return scheduler
