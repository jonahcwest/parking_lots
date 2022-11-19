import torch
from torch.utils.data import DataLoader, Dataset

from main import MODEL_STATE, Data, Net


def main():
    device = torch.device("mps")

    data = Data("dataset/test")
    data_loader = DataLoader(data, batch_size=8)

    net = Net()
    net = net.to(device)

    net.load_state_dict(torch.load(MODEL_STATE))

    inputs, labels = next(iter(data_loader))
    inputs = inputs.to(device)
    outputs = net(inputs)

    for x, y in zip(outputs, labels):
        print(round(x[0].item()), round(x[1].item()))
        print(round(y[0].item()), round(y[1].item()))
        print("===")


if __name__ == "__main__":
    main()
