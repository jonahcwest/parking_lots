import itertools
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MODEL_STATE = "model.pt"
OPTIMIZER_STATE = "optimizer.pt"

MAX_IMAGES = 1000
BATCH_SIZE = 32
LOG_INTERVAL = 5


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear = nn.Linear(394384, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear(x))
        return x


class Data(Dataset):
    def __init__(self, data_path):
        parsed = 0
        with open(f"{data_path}/_annotations.coco.json", "r") as f:
            content = f.read()
            parsed = json.loads(content)

        id_to_file_name = {}
        for image in parsed["images"]:
            id_to_file_name[image["id"]] = image["file_name"]

        file_name_to_spots = {}
        for a in parsed["annotations"]:
            file_name = id_to_file_name[a["image_id"]]
            category_id = a["category_id"]

            if not file_name in file_name_to_spots:
                file_name_to_spots[file_name] = [0, 0, 0]

            file_name_to_spots[file_name][category_id] += 1

        # choose MAX_IMAGES random images
        file_name_to_spots = dict(
            random.sample(list(file_name_to_spots.items()), MAX_IMAGES)
        )

        self.images = []
        for file_name in file_name_to_spots:
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            with Image.open(f"{data_path}/{file_name}", "r") as image:
                self.images.append(
                    [
                        normalize(to_tensor(image)),
                        torch.Tensor([*file_name_to_spots[file_name][1:]]),
                    ]
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def main():
    device = torch.device("mps")

    net = Net()
    net = net.to(device)
    if os.path.isfile(MODEL_STATE):
        net.load_state_dict(torch.load(MODEL_STATE))

    optimizer = optim.Adam(net.parameters())
    if os.path.isfile(OPTIMIZER_STATE):
        optimizer.load_state_dict(torch.load(OPTIMIZER_STATE))

    criterion = nn.MSELoss()

    data = Data("dataset/train")
    data_loader = DataLoader(data, batch_size=BATCH_SIZE)

    elapsed = time.time()
    elapsed_steps = 0
    running_loss = 0

    while True:
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            elapsed_steps += 1
            running_loss += loss.item()

            if time.time() - elapsed > LOG_INTERVAL:
                torch.save(net.state_dict(), MODEL_STATE)
                torch.save(optimizer.state_dict(), OPTIMIZER_STATE)

                print(f"avg loss: {running_loss / elapsed_steps}")

                # print a single sample as an example
                print("example:")
                print(round(outputs[0][0].item()), round(outputs[0][1].item()))
                print(round(labels[0][0].item()), round(labels[0][1].item()))

                elapsed = time.time()
                elapsed_steps = 0
                running_loss = 0


if __name__ == "__main__":
    main()
