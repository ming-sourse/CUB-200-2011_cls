import torchvision.models as models
from torch import nn

def get_model(num_classes=200, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
