import torchvision.transforms.functional as TF


def rotate_image(image, angle):
    """
    画像を回転させる

    Parameters
    ----------
    image : torch.Tensor
        (N, C, H, W)

    angle : float
        回転角度（度）

    Returns
    -------
    rotated_image : torch.Tensor
    """

    rotated_image = TF.rotate(
        image,
        angle=angle
    )

    return rotated_image