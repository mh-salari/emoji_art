import json
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_keys_from_value(d, val):
    key = [k for k, v in d.items() if v == val][0]
    return key


def plot_samples(dataset, mean, std):
    figure = plt.figure(figsize=(4, 4), facecolor="none")
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        inp = img.numpy().transpose((1, 2, 0))
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

        figure.add_subplot(rows, cols, i)
        plt.title(get_keys_from_value(dataset.class_to_idx, label))
        plt.axis("off")
        plt.imshow(inp)
    plt.show()


def load_data(dataset_path, transforms, batch_size):
    dataset = ImageFolder(root=dataset_path, transform=transforms)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    return dataset, dataloader


def make_model(num_classes):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model


def train_loop(model, num_epochs, optimizer, loss_fn, dataloader, dataset_size, device):
    model.train()
    print(f"Training the model for {num_epochs} epochs")
    for epoch in tqdm(range(1, num_epochs + 1)):  # loop over the dataset multiple times

        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if epoch == 1 or epoch == num_epochs or epoch % 5 == 0:
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            tqdm.write(f"epoch:{epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("Finished Training")
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

    dataset, dataloader = load_data(dataset_path, train_transforms, 512)
    # plot_samples(dataset, mean, std)
    model = make_model(len(dataset.classes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trained_model = train_loop(
        model=model,
        num_epochs=40,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dataloader=dataloader,
        dataset_size=len(dataset),
        device=device,
    )

    torch.save(
        trained_model.state_dict(), data_path / f"model_{emojis_size}{emojis_size}.pt"
    )
    with open(data_path / "class_to_idx.json", "w") as outfile:
        json.dump(dataset.class_to_idx, outfile)
