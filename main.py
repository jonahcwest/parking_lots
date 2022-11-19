import itertools
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import torchvision
import torchvision.transforms as transforms


DATA_PATH = "dataset/train"
MAX_IMAGES = 200


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
                    {
                        "image": normalize(to_tensor(image)),
                        "spots": file_name_to_spots[file_name],
                    }
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def main():
    data = Data()
    data_iter = iter(data)


if __name__ == "__main__":
    main()
