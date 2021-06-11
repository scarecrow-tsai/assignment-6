import os
import random
from utils.test_loop import test_loop
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.load_model import ModelCL
from utils.train_loop import train_loop
from utils.load_data import load_dataset, create_samplers


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(100)


################################
## CONFIG
################################
DATASET_NAME = "mnist"
DATASET_ROOT_PATH = f"./../data/{DATASET_NAME}/"
NUM_CLASSES = 10


IMAGE_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001


# SET GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nWe're using =>", device)


################################
## LOAD DATASET
################################

image_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
    "test": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
}

# datasets
train_dataset = load_dataset(
    dataset_name=DATASET_NAME,
    dataset_root_path=DATASET_ROOT_PATH,
    is_train=True,
    image_transforms=image_transforms["train"],
)

test_dataset = load_dataset(
    dataset_name=DATASET_NAME,
    dataset_root_path=DATASET_ROOT_PATH,
    is_train=False,
    image_transforms=image_transforms["test"],
)

# train-val sampler
train_sampler, val_sampler = create_samplers(train_dataset, 0.8)


# dataloader
train_loader = DataLoader(
    dataset=train_dataset, shuffle=False, batch_size=BATCH_SIZE, sampler=train_sampler,
)

val_loader = DataLoader(
    dataset=train_dataset, shuffle=False, batch_size=1, sampler=val_sampler
)

test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

## Data Sanity Check
print(f"\nTrain loader = {next(iter(train_loader))[0].shape}")
print(f"Val loader = {next(iter(val_loader))[0].shape}")
print(f"Test loader = {next(iter(test_loader))[0].shape}")
print(f"\nTrain loader length = {len(train_loader)}")
print(f"Val loader length = {len(val_loader)}")
print(f"Test loader length = {len(test_loader)}")


################################
## LOAD MODEL
################################
model = ModelCL(num_classes=NUM_CLASSES, norm_type="bnorm")

x_train_example, y_train_example = next(iter(train_loader))
y_pred_example = model(x_train_example)

print("\nShape of output pred = ", y_pred_example.shape)

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

################################
## Train Loop
################################
trained_model, loss_stats, acc_stats = train_loop(
    model=model,
    epochs=EPOCHS,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
)


################################
## Test Loop
################################
y_pred_list, y_true_list = test_loop(
    model=trained_model, test_loader=test_loader, device=device,
)
