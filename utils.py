import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from PIL import Image

import os

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(PROJ_ROOT, 'TMP')
ROOMS_DIR = os.path.join(PROJ_ROOT, 'rooms')


class RoomDeterminant:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classifier = self._create_classifier()
       
    def _create_classifier(self):
        resnet18 = models.resnet18(pretrained = True)

        classifier = nn.Sequential(
            resnet18,
            nn.ReLU(inplace = True),
            nn.Linear(1000, 2),
            nn.Sigmoid()
        )

        classifier.to(self.device)
        classifier.load_state_dict(torch.load('classifier.pt', map_location = self.device))
        return classifier

    def determine(self, images):
        '''
        in: images - list, список путей к фотографиям
        out: 0 or 1 (0 - doesn't match, 1 - matchs)
        '''
        self.classifier.eval()
        images = pd.Series(images)
        # return images
        images = images.apply(lambda elem: Image.open(elem))
        images = images.apply(lambda elem: elem.convert('RGB'))
        transform = transforms.Compose([
                        transforms.Resize((620, 620)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomRotation(30),
                        transforms.ToTensor(),
                        # Use mean and std for pretrained models
                        # https://pytorch.org/docs/stable/torchvision/models.html
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                ])
        images = images.apply(lambda elem: transform(elem))
        images = torch.stack(tuple(images)).view(3, 620*5, 620*3)

        pred = self.classifier(images.to(self.device).unsqueeze(0))
        pred = torch.argmax(pred, axis = 1)
        return int(pred)


def get_room_posibilities(target_name: str) -> dict:
    determinant = RoomDeterminant()
    room_images = {}
    posibilities = {}

    target_full_path = os.path.join(TMP_DIR, target_name)

    dirname, roomnames, _ = os.walk(ROOMS_DIR)
    for roomname in roomnames:
        _, _, room_images = os.walk(os.path.join(dirname,roomname))
        images = [target_full_path]
        images += room_images
        room_images[roomname] = images

    for room_name, images in room_images.items():
        posibilities[room_name] = determinant.determine(images)

    return posibilities




