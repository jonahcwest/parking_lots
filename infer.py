from argparse import ArgumentParser
from PIL import Image

import torch
import torchvision.transforms as transforms

from main import MODEL_STATE, Net


def main():

    net = Net()
    net.load_state_dict(torch.load(MODEL_STATE))

    parser = ArgumentParser()
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    inputs = 0
    with Image.open(args.file, "r") as image:
        inputs = normalize(to_tensor(image))
    inputs = inputs[None, :]

    outputs = net(inputs)

    print("empty:", round(outputs[0][0].item()))
    print("occupied:", round(outputs[0][1].item()))


if __name__ == "__main__":
    main()
