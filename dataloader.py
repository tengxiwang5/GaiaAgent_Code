import os
import imageio as iio
import tifffile as tiff
from torch.utils.data import Dataset as BaseDataset

def center_crop(imm, size, imtype = 'image'):
    h = int(size[0]/2)
    w = int(size[1]/2)
    ch = int(imm.shape[0]/2)
    cw = int(imm.shape[1]/2)
    if imtype == 'image':
        return imm[ch-h:ch+h, cw-w:cw+w, :]
    else:
        return imm[ch-h:ch+h, cw-w:cw+w]
class Dataset(BaseDataset):
    """Read images and masks; load text only for training."""

    def __init__(
            self, 
            root,
            text_dir=None,
            mode="train",  # 设置模式   
            augmentation=False,
            max_text_length=150,  # 设置最大文本长度
            pad_token=" "  # 设置填充字符（如果需要）
    ):
        self.t1_images_dir = os.path.join(root, 'p1/')
        self.t2_images_dir = os.path.join(root, 'p2/')
        self.masks2d_dir = os.path.join(root, '2d/')
        self.masks3d_dir = os.path.join(root, '3d/')
        self.texts_dir = os.path.join(text_dir or root, 'text/') if mode == "train" else None
        self.mode = mode
        self.ids = os.listdir(self.t1_images_dir)
        self.idm = os.listdir(self.masks3d_dir)

        self.t1_images_fps = [os.path.join(self.t1_images_dir, image_id) for image_id in self.ids]
        self.t2_images_fps = [os.path.join(self.t2_images_dir, image_id) for image_id in self.ids]
        self.masks2d_fps = [os.path.join(self.masks2d_dir, image_id) for image_id in self.ids]
        self.masks3d_fps = [os.path.join(self.masks3d_dir, image_id) for image_id in self.idm]
        self.texts_fps = (
            [os.path.join(self.texts_dir, image_id.replace('.tif', '.txt')) for image_id in self.ids]
            if self.mode == "train"
            else None
        )
        
        self.augmentation = augmentation
        self.max_text_length = 150  # 设定文本最大长度
        self.pad_token = pad_token  # 设定填充值

    def __getitem__(self, i):
        # 读取影像数据
        t1 = iio.imread(self.t1_images_fps[i])
        t2 = iio.imread(self.t2_images_fps[i])
        mask2d = iio.imread(self.masks2d_fps[i])
        mask3d = tiff.imread(self.masks3d_fps[i])

        # print(f"t1 shape: {t1.shape}, t2 shape: {t2.shape}")
        if t2.shape[-1] == 4:
            t2 = t2[:, :, :3]  # 只取 RGB 通道
        if t1.shape[-1] == 4:
            t1 = t1[:, :, :3]  # 只取 RGB 通道


        # 处理数据增强
        if self.augmentation:
            sample = self.augmentation(image=t1, t2=t2, mask=mask2d, mask3d=mask3d)
            t1, t2, mask2d, mask3d = sample['image'], sample['t2'], sample['mask'], sample['mask3d']

        if self.mode != "train":
            return t1, t2, mask2d, mask3d

        text = ""
        text_path = self.texts_fps[i]
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        else:
            text = text.ljust(self.max_text_length, self.pad_token)

        return text, t1, t2, mask2d, mask3d

    def __len__(self):
        return len(self.ids)




