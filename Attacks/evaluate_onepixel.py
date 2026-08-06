import time
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchattacks
import torchvision.utils as vutils

from Transforms.rotation import rotate_image
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
baseline_success = 0
rotation_success = 0
total_time = 0.0
NoAttacks = 0
save_count = 0
attack_count = 0

MAX_IMAGES = 100

SAVE_DIR = "Result/Rot5"

def create_difference(original, adversarial):

    diff = torch.abs(adversarial - original)

    diff = diff / diff.max()

    return diff

for images, labels in testloader:

    if total >= MAX_IMAGES:
        break
    total += 1

    images = images.to(device)
    labels = labels.to(device)

    # 攻撃前予測
    with torch.no_grad():
        before = attack_model(images).argmax(dim=1)

    # 元々誤分類している画像は除外
    if before.item() != labels.item():
        NoAttacks += 1
        continue

    start = time.time()

    attack_count += 1

    adv_images = attack(images, labels)
    #回転前の攻撃成功カウント
    with torch.no_grad():
        before_rotate = attack_model(adv_images).argmax(dim=1)

    baseline_ok = (before_rotate.item() != labels.item())
    if baseline_ok:
        baseline_success += 1
    if not baseline_ok:
        result = "Baseline Failed"
        print(
            f"[{total:3d}] "
            f"Label:{labels.item()} "
            f"Before:{before.item()} "
            f"Result:{result}"
        )
        continue
   
    #回転処理
    rotated_images = rotate_image(
        adv_images,
        angle=5
    )

    adv_images = rotated_images
   
    end = time.time()

    total_time += (end - start)

    # 攻撃後予測
    with torch.no_grad():
        after = attack_model(adv_images).argmax(dim=1)

    if baseline_ok:

        if after.item() != labels.item():
            rotation_success += 1
            save_count += 1
            result = "Success"

            diff = create_difference(images, adv_images)

            vutils.save_image(
                images.cpu(),
                f"{SAVE_DIR}/original_{save_count:03d}.png"
            )
            vutils.save_image(
                adv_images.cpu(),
                f"{SAVE_DIR}/adversarial_{save_count:03d}.png"
            )
            vutils.save_image(
                diff.cpu(),
                f"{SAVE_DIR}/difference_{save_count:03d}.png"
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


retention = 100 * rotation_success / baseline_success

# ==========================
# Result
# ==========================
print()
print("=" * 40)
print(f"Images          : {total}")
print(f"NoAttacks          : {NoAttacks}")
print(f"Baseline Success : {baseline_success}")
print(f"Rotation Success : {rotation_success}")
print(f"Attack Retention : {retention:.2f}%")
print(f"Average Time : {total_time / attack_count:.2f} sec/image")
print("=" * 40)