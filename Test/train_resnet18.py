import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from DataSet.cifar10 import get_trainloader, get_testloader


# -----------------------------
# 学習1epoch分
# -----------------------------
def train_one_epoch(model, trainloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        # 勾配リセット
        optimizer.zero_grad()

        # 順伝播
        outputs = model(images)

        # 損失計算
        loss = criterion(outputs, labels)

        # 逆伝播
        loss.backward()

        # 更新
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)

    return avg_loss



# -----------------------------
# 評価
# -----------------------------
def evaluate(model, testloader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total

    return accuracy



# -----------------------------
# main
# -----------------------------
def main():

    # データ
    trainloader = get_trainloader(batch_size=64)
    testloader = get_testloader(batch_size=100)


    # GPU/CPU設定
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)


    # ResNet18
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    model = model.to(device)


    # 損失関数
    criterion = nn.CrossEntropyLoss()


    # 最適化手法
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )


    # 学習設定
    epochs = 50


    # 最高精度保存用
    best_accuracy = 0.0


    # -----------------------------
    # 学習開始
    # -----------------------------
    for epoch in range(epochs):

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


        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {test_accuracy:.2f}%"
        )


        # 精度が更新されたら保存
        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            torch.save(
                model.state_dict(),
                "Models/resnet18_cifar10_best.pth"
            )

            print("  → Best model saved")


    print("Training finished")
    print(
        f"Best Accuracy: {best_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()