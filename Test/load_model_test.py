import torch
import torch.nn as nn
import torchvision.models as models

from DataSet.cifar10 import get_testloader


# -----------------------------
# 評価関数
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

    # デバイス設定
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)


    # テストデータ
    testloader = get_testloader(
        batch_size=100
    )


    # ResNet18作成
    model = models.resnet18(
        weights=None
    )

    # CIFAR-10用に変更
    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )


    # 学習済みモデル読み込み
    model.load_state_dict(
        torch.load(
            "Models/resnet18_cifar10_best.pth",
            map_location=device
        )
    )


    # GPU/CPUへ移動
    model = model.to(device)


    print("Model loaded successfully")


    # 精度評価
    accuracy = evaluate(
        model,
        testloader,
        device
    )


    print(
        f"Test Accuracy : {accuracy:.2f}%"
    )



if __name__ == "__main__":
    main()