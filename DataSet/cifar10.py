import torch
import torchvision
import torchvision.transforms as transforms

"""学習用データのダウンロードと受け渡し"""
def get_trainloader(batch_size = 64):

    # 前処理
    Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # データセット読み込み
    TrainDataSet = torchvision.datasets.CIFAR10(
        root = './DataSet/CIFAR10',
        train = True,
        download = True,
        transform = Transform
    )

    #バッチ分割
    trainloader = torch.utils.data.DataLoader(
        TrainDataSet,
        batch_size=batch_size,
        shuffle = True,
        num_workers=0
    )
    return trainloader

"""テスト用データのダウンロードと受け渡し"""
def get_testloader(batch_size = 100):

    # 前処理
    Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # データセット読み込み
    TestDataSet = torchvision.datasets.CIFAR10(
        root = './DataSet/CIFAR10',
        train = False,
        download = True,
        transform = Transform
    )

    #バッチ分割
    testloader = torch.utils.data.DataLoader(
        TestDataSet,
        batch_size = batch_size,
        shuffle = False,
        num_workers=0
    )
    return testloader