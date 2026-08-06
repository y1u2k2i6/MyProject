import torch
import torch.nn as nn
import torchvision.models as models

from DataSet.cifar10 import get_testloader
from Transforms.rotation import rotate_image

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------
# モデル読み込み
# ------------------------

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(
    torch.load(
        "Models/resnet18_cifar10_best.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded")

# ------------------------
# データ
# ------------------------

testloader = get_testloader(batch_size=1)

correct = 0
total = 0

ANGLE = 20

# ------------------------
# 評価
# ------------------------

with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        # 回転
        images = rotate_image(
            images,
            angle=ANGLE
        )

        outputs = model(images)

        pred = outputs.argmax(dim=1)

        total += 1

        if pred.item() == labels.item():
            correct += 1

accuracy = 100 * correct / total

print(f"Rotate : {ANGLE}°")
print(f"Accuracy : {accuracy:.2f}%")