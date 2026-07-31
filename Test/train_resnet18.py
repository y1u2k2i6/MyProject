#-------------------
#resnet18のモデル学習
#-------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from DataSet.cifar10 import get_trainloader, get_testloader

#ハイパーパラメータ-----
BATCH_SIZE = 64
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
#---------------------

#データセット読み込み--------------------------
trainloader = get_trainloader(batch_size=64)
testloader = get_testloader(batch_size=100)
#--------------------------------------------

#モデル作成(全結合層をCIFAR10用に変更｛1000クラス→10クラス｝)
model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features,
    10
)
#--------------------------------------------------------

#デバイス割り当て-------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = model.to(device)
#---------------------------

#損失関数--------------------------
criterion = nn.CrossEntropyLoss()
#---------------------------------

#最適化------------------------
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)
#-----------------------------

print(model)

def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    average_loss = running_loss / len(trainloader)
    return average_loss


def evaluate(model, testloader, device):

    # 評価モード
    model.eval()

    # 正解数
    correct = 0

    # 全データ数
    total = 0

    # 勾配計算を無効化
    with torch.no_grad():

        for images, labels in testloader:

            # CPU/GPUへ転送
            images = images.to(device)
            labels = labels.to(device)

            # 推論
            outputs = model(images)

            # 最も確率の高いクラスを取得
            _, predicted = torch.max(outputs, 1)

            # データ数を加算
            total += labels.size(0)

            # 正解数を加算
            correct += (predicted == labels).sum().item()

    # 正解率(%)
    accuracy = 100 * correct / total

    return accuracy


train_loss = train_one_epoch(
    model,
    trainloader,
    criterion,
    optimizer,
    device
)

test_accuracy = evaluate(
    model,
    testloader,
    device
)

print(f"Train Loss : {train_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.2f}%")