import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import ray
from ray.util.sgd.torch import TorchTrainer
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
from ray.util.sgd.torch.resnet import ResNet18


def cifar_creator(config):
    """Returns dataloaders to be used in `train` and `validate`."""
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])  # meanstd transformation
    train_loader = DataLoader(
        CIFAR10(root="~/data", download=True, transform=tfms), batch_size=config["batch"])
    validation_loader = DataLoader(
        CIFAR10(root="~/data", download=True, transform=tfms), batch_size=config["batch"])
    return train_loader, validation_loader


def optimizer_creator(model, config):
    """Returns an optimizer (or multiple)"""
    return torch.optim.SGD(model.parameters(), lr=config["lr"])

ray.init(address="auto")

trainer = TorchTrainer(
    model_creator=ResNet18,  # A function that returns a nn.Module
    data_creator=cifar_creator,  # A function that returns dataloaders
    optimizer_creator=optimizer_creator,  # A function that returns an optimizer
    loss_creator=torch.nn.CrossEntropyLoss,  # A loss function
    config={"lr": 0.01, "batch": 64},  # parameters
    num_workers=4,  # amount of parallelism
    use_gpu=torch.cuda.is_available(),
    use_tqdm=True)

stats = trainer.train()
print(trainer.validate())

torch.save(trainer.state_dict(), "checkpoint.pt")
trainer.shutdown()
print("success!")
