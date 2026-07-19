import albumentations as albu
import albumentations.pytorch as apt

def get_training_augmentations(m=[0, 0, 0], s=[1, 1, 1], image_size=1024):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.GaussNoise(p=0.2),

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),

        # 确保所有输入图像和掩码尺寸一致
        albu.Resize(image_size, image_size, p=1), 

        albu.Normalize(mean=m, std=s),
        apt.ToTensorV2(),
    ]
    return albu.Compose(train_transform, additional_targets={'t2': 'image', 'mask3d': 'mask'})


def get_validation_augmentations(m=[0, 0, 0], s=[1, 1, 1], image_size=400):
    val_transform = [
        # 确保验证时图像和掩码尺寸一致
        # albu.Resize(image_size, image_size, p=1),

        albu.Normalize(mean=m, std=s),
        apt.ToTensorV2(),
    ]

    return albu.Compose(val_transform, additional_targets={'t2': 'image', 'mask3d': 'mask'})
