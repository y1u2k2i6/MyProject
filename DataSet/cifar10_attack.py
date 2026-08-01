import torch
import torchvision
import torchvision.transforms as transforms


def get_attackloader(batch_size=1):

    Transform = transforms.Compose([
        transforms.ToTensor()
    ])


    TestDataSet = torchvision.datasets.CIFAR10(
        root='./DataSet/CIFAR10',
        train=False,
        download=False,
        transform=Transform
    )


    loader = torch.utils.data.DataLoader(
        TestDataSet,
        batch_size=batch_size,
        shuffle=False
    )

    return loader