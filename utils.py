import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from uuid import uuid4

import os

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(PROJ_ROOT, 'TMP')
ROOMS_DIR = os.path.join(PROJ_ROOT, 'rooms')


def save_target_image(image):
    Path(TMP_DIR).mkdir(exist_ok=True)
    image.filename = str(uuid4()) + '.jpg'
    image.save(os.path.join(TMP_DIR, image.filename))


class RoomDeterminant:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classifier = self._create_classifier()

    def _create_classifier(self):
        resnet18 = models.resnet18(pretrained=True)

        classifier = nn.Sequential(
            resnet18,
            nn.ReLU(inplace=True),
            nn.Linear(1000, 2),
            nn.Sigmoid()
        )

        classifier.to(self.device)
        classifier.load_state_dict(torch.load(
            'classifier.pt', map_location=self.device))
        return classifier

    def get_room_possibilities(self, target_name: str) -> dict:
        """
        Args:
            target_name: str - name of a target file
        Returns:
            dict of possibilities:
            {
                room-name : bool,
                ...
            }
        """
        possibilities = {}
        target_full_path = os.path.join(TMP_DIR, target_name)

        rooms_images = self._extract_rooms_images()

        for room_name, images in rooms_images.items():
            possibilities[room_name] = self._is_correct_room(
                images,
                target_full_path,
            )

        return possibilities

    @staticmethod
    def _extract_rooms_images():
        """ Extracts room images from 'rooms/' directory """
        rooms_images = {}
        _, roomnames, _ = next(os.walk(ROOMS_DIR))
        for roomname in roomnames:
            room_dir, _, room_images = next(
                os.walk(os.path.join(ROOMS_DIR, roomname)))

            images = [os.path.join(room_dir, im) for im in room_images]
            rooms_images[roomname] = images
        return rooms_images

    def _is_correct_room(self, images, target) -> bool:
        """
        Args:
            images: list[str], paths to room photos
            target: str - path to target
        Returns:
            bool: weather room mathc target

        """
        images = [target] + images
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
        pred = torch.argmax(pred, axis=1)
        return bool(pred)




