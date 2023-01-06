import json
from torchvision import transforms
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch import nn
from PIL import Image
from patchify import patchify, unpatchify
from tqdm.auto import tqdm


def get_keys_from_value(d, val):
    key = [k for k, v in d.items() if v == val][0]
    return key


def make_model(num_classes):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model


if __name__ == "__main__":

    emojis_size = 10

    data_path = Path("/media/hue/Data/codes/conv_emoji/data/")
    dataset_path = data_path / f"{emojis_size}x{emojis_size}"

    mean = np.array([0.9778, 1.0140, 1.0329])
    std = np.array([1.1388, 1.1836, 1.3645])

    train_transforms = transforms.Compose(
        [
            transforms.Resize(size=(emojis_size, emojis_size)),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.9)),
        ]
    )

    with open(data_path / "class_to_idx.json", "r") as fp:
        class_idx = json.load(fp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = make_model(len(class_idx))
    model.load_state_dict(
        torch.load(data_path / f"model_{emojis_size}{emojis_size}.pt")
    )
    model = model.to(device)
    image = Image.open("ha.jpg").convert("RGB")

    image = np.asarray(image)

    # splitting the image into patches
    image_height, image_width, channel_count = image.shape
    patch_height, patch_width, step = emojis_size, emojis_size, emojis_size
    patch_shape = (patch_height, patch_width, channel_count)
    patches = patchify(image, patch_shape, step=step)
    print(patches.shape)
    output_patches = patches.copy()

    patches_tensor = torch.tensor(patches, dtype=torch.float32).squeeze(dim=2)
    patches_tensor = patches_tensor.permute(0, 1, 4, 2, 3)

    patches_tensor = patches_tensor / 255

    print(patches_tensor.shape)

    model.eval()

    pred_transforms = transforms.Compose(
        [
            transforms.Resize(size=(emojis_size, emojis_size)),
            transforms.Normalize(
                mean=[0.9778, 1.0140, 1.0329], std=[1.1388, 1.1836, 1.3645]
            ),
        ]
    )

    with torch.inference_mode():
        for i in tqdm(range(patches_tensor.shape[0])):
            batch = patches_tensor[i]
            patch_transformed = pred_transforms(batch)
            pred = model(patch_transformed.to(device))
            probs = torch.softmax(pred, dim=1)
            pred_label = torch.argmax(probs, dim=1)
            emoji_class = [
                get_keys_from_value(class_idx, label.cpu().item())
                for label in pred_label
            ]

            for column, emoji in enumerate(emoji_class):
                img = Image.open(
                    f"/media/hue/Data/codes/conv_emoji/data/{emojis_size}x{emojis_size}/{emoji}/1.jpg"
                )
                output_patches[i][column] = np.asarray(img)
    # merging back patches
    output_height = image_height - (image_height - patch_height) % step
    output_width = image_width - (image_width - patch_width) % step
    output_shape = (output_height, output_width, channel_count)
    output_image = unpatchify(output_patches, output_shape)
    output_image = Image.fromarray(output_image)
    output_image.save(f"output_{emojis_size}.jpg")
    output_image
