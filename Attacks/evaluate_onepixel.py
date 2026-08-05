import time
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchattacks
import torchvision.utils as vutils

from DataSet.cifar10_attack import get_attackloader
from Attacks.normalize_model import NormalizeModel


# ==========================
# Device
# ==========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================
# Model
# ==========================
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

attack_model = NormalizeModel(model)
attack_model = attack_model.to(device)
attack_model.eval()

print("Model loaded")


# ==========================
# DataLoader
# ==========================
testloader = get_attackloader(batch_size=1)


# ==========================
# Attack
# ==========================
attack = torchattacks.OnePixel(
    attack_model,
    pixels=1,
    steps=50,
    popsize=20
)


# ==========================
# Evaluation
# ==========================
total = 0
success = 0
total_time = 0.0

MAX_IMAGES = 100

SAVE_DIR = "Result/Baseline"

def create_difference(original, adversarial):

    diff = torch.abs(adversarial - original)

    diff = diff / diff.max()

    return diff

for images, labels in testloader:

    if total >= MAX_IMAGES:
        break

    images = images.to(device)
    labels = labels.to(device)

    # 攻撃前予測
    with torch.no_grad():
        before = attack_model(images).argmax(dim=1)

    # 元々誤分類している画像は除外
    if before.item() != labels.item():
        continue

    start = time.time()

    adv_images = attack(images, labels)

    end = time.time()

    total_time += (end - start)

    # 攻撃後予測
    with torch.no_grad():
        after = attack_model(adv_images).argmax(dim=1)

    total += 1

    if after.item() != labels.item():
        success += 1
        result = "Success"

        diff = create_difference(images, adv_images)

        vutils.save_image(
            images.cpu(),
            f"{SAVE_DIR}/original_{success:03d}.png"
        )
        vutils.save_image(
            adv_images.cpu(),
            f"{SAVE_DIR}/adversarial_{success:03d}.png"
        )
        vutils.save_image(
            diff.cpu(),
            f"{SAVE_DIR}/difference_{success:03d}.png"
        )

    else:
        result = "Failed"

    print(
        f"[{total:3d}] "
        f"Label:{labels.item()} "
        f"Before:{before.item()} "
        f"After:{after.item()} "
        f"{result}"
    )


# ==========================
# Result
# ==========================
print()
print("=" * 40)
print(f"Images          : {total}")
print(f"Attack Success  : {success}")
print(f"Success Rate    : {100 * success / total:.2f}%")
print(f"Average Time    : {total_time / total:.2f} sec/image")
print("=" * 40)