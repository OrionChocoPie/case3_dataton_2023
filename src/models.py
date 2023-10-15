import torch.nn as nn
from torchvision import models


def get_model(model_type, device, num_classes, freeze=False):
    if model_type == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_type == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_type == "efficientnet_b1":
        model = models.efficientnet_b1(weights="IMAGENET1K_V2")
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_type == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise Exception("Wrong model type")

    model = model.to(device)
    return model
