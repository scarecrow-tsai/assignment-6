import numpy as np

from torchvision import datasets
from torch.utils.data import SubsetRandomSampler


def mnist_dataset(dataset_root_path, is_train, image_transforms):
    return datasets.MNIST(
        root=dataset_root_path,
        train=is_train,
        download=True,
        transform=image_transforms,
    )


def load_dataset(dataset_name, *args, **kwargs):
    if dataset_name == "mnist":
        return mnist_dataset(*args, **kwargs)
    else:
        raise ValueError("Incorrect dataset name.")


def create_samplers(dataset, train_percent):
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))

    np.random.shuffle(dataset_indices)

    train_split_index = int(np.floor(train_percent * dataset_size))

    train_idx = dataset_indices[:train_split_index]
    val_idx = dataset_indices[train_split_index:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return train_sampler, val_sampler


if __name__ == "__main__":
    pass
