import torch
import torch.nn as nn
import torchvision.models as models
import torchattacks

from DataSet.cifar10 import get_testloader
from DataSet.cifar10_attack import get_attackloader
from Attacks.normalize_model import NormalizeModel


# -------------------------
# デバイス
# -------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# -------------------------
# CIFAR-10 test data
# -------------------------
testloader = get_attackloader(
    batch_size=1
)


# 1枚取得
images, labels = next(iter(testloader))

images = images.to(device)
labels = labels.to(device)


# -------------------------
# ResNet18
# -------------------------
model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)


model.load_state_dict(
    torch.load(
        "Models/resnet18_cifar10_best.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# One Pixel Attack用モデル
attack_model = NormalizeModel(model)
attack_model = attack_model.to(device)
attack_model.eval()


print("Model loaded")


# -------------------------
# 攻撃前の予測
# -------------------------
with torch.no_grad():

    output = attack_model(images)

    pred = output.argmax(dim=1)


print("True label :", labels.item())
print("Before attack prediction :", pred.item())


# -------------------------
# One Pixel Attack
# -------------------------

attack = torchattacks.OnePixel(
    attack_model,
    pixels=1,
    steps=50,
    popsize=20
)


adv_images = attack(
    images,
    labels
)


# -------------------------
# 攻撃後予測
# -------------------------

with torch.no_grad():

    output = model(adv_images)

    adv_pred = output.argmax(dim=1)


print("After attack prediction :", adv_pred.item())

print(images.min())
print(images.max())
if pred.item() != adv_pred.item():

    print("Attack Success!")

else:

    print("Attack Failed")