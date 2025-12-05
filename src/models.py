import torch
import torch.nn as nn
import timm


class PretrainedBackbone(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=7, pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class SmallCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(name='resnet50', num_classes=7, pretrained=True, dropout=0.5):
    name = name.lower()
    if name in ('resnet50', 'resnet18', 'efficientnet_b0', 'mobilenetv3_small'):
        return PretrainedBackbone(model_name=name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif name in ('smallcnn', 'baseline'):
        return SmallCNN(in_channels=3, num_classes=num_classes)
    else:
        # fallback to timm generic
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        return model
