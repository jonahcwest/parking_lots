import itertools
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DATA_PATH = "dataset/train"
MAX_IMAGES = 1000
BATCH_SIZE = 32
LOG_INTERVAL = 3


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(606744, 2)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class Data(Dataset):
    def __init__(self):
        parsed = 0
        with open(f"{DATA_PATH}/_annotations.coco.json", "r") as f:
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

        file_name_to_spots = dict(
            itertools.islice(file_name_to_spots.items(), MAX_IMAGES)
        )

        self.images = []
        for file_name in file_name_to_spots:
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            with Image.open(f"{DATA_PATH}/{file_name}", "r") as image:
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

    data = Data()
    data_loader = DataLoader(data, batch_size=BATCH_SIZE)

    net = Net()
    net = net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    elapsed = time.time()
    elapsed_steps = 0

    while True:
        running_loss = 0.0
        for data in data_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            elapsed_steps += 1
            running_loss += loss.item()

            if time.time() - elapsed > LOG_INTERVAL:
                print(f"avg loss: {running_loss / elapsed_steps}")

                # print examples
                print("example:")
                print(round(outputs[0][0].item()), round(outputs[0][1].item()))
                print(round(labels[0][0].item()), round(labels[0][1].item()))

                elapsed = time.time()
                elapsed_steps = 0
                running_loss = 0


if __name__ == "__main__":
    main()
