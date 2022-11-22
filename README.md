# Usage

Note: To train on non-Apple Silicon platforms, replace

    device = torch.device("mps")

with

    device = torch.device("cpu")

## Dependencies

- PyTorch
- TorchVision
- PIL

## Parameters

- `MAX_IMAGES`: Maximum number of images loaded into memory during training
- `LOG_INTERVAL`: Seconds between each log to stdout

## Training

    python3 main.py

Model state will be saved in `model.pt` and optimizer state will be saved in `optimizer.pt`.

The program will output the average loss over the past LOG_INTERVAL seconds and a random example of the model's prediction and the truth value.

## Inference

    python3 infer.py $IMAGE

The model will output the number of empty and occupied parking spots.

# Limitations

- Currently, the model can only take 640x640 images.
