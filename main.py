from torch.utils.data import Dataset
import json


class Data(Dataset):
    def __init__(self):
        parsed = 0
        with open("dataset/train/_annotations.coco.json", "r") as f:
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


Data()
