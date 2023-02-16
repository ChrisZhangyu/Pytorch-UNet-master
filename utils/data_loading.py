import logging
import os
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext == '.nii':
        return nib.load(filename).get_fdata()
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    # glob函数是搜索函数，返回所有匹配的文件路径列表，参数是文件的路径
    temp = list(mask_dir.glob(idx + mask_suffix + '.*'))
    mask_file = temp[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        # 输入图片的路径
        self.images_dir = Path(images_dir)
        # groundTruth的路径
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        # 表示图片的缩放比例
        self.scale = scale
        # 待定
        self.mask_suffix = mask_suffix
        # ids表示每张图片的名字构成的列表
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        # 如果是空说明文件路径中没有图片
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        # 进程池，多进程并行处理任务，提高速度
        with Pool() as p:
            # tqdm是在控制台显示进度条
            unique = list(tqdm(
                # p.imap()利用进程池中的进程调用方法，
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        if type(pil_img) is np.memmap:
            w, h = pil_img.shape[0], pil_img.shape[1]
        else:
            w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            # mask_values表示mask的有几种
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class AppendFilterDataSet(BasicDataset):
    def __init__(self, images_dir, mask_dir, filter_dir, scale=1):
        super(AppendFilterDataSet, self).__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
        self.filter_dir = filter_dir

    def __getitem__(self, idx):
        image_and_mask = super().__getitem__(idx)
        name = self.ids[idx]
        filter_file = list(self.images_dir.glob(name + '.*'))
        assert len(filter_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {filter_file}'
        filter_img = load_image(filter_file[0])
        filter_img = super().preprocess(self.mask_values, filter_img, self.scale, is_mask=False)
        return {
            'image': image_and_mask['image'],
            'mask': image_and_mask['mask'],
            'filter': torch.as_tensor(filter_img.copy()).float().contiguous(),
        }


if __name__ == '__main__':
    image_dir = "../data/imgs"
    mask_dir = "../data/masks"
    filter_dir = "../data/filter"
    from torch.utils.data import DataLoader
    testDataSet = AppendFilterDataSet(image_dir, mask_dir, filter_dir)
    train_loader = DataLoader(testDataSet, shuffle=True)
    for batch in train_loader:
        print(np.squeeze(batch['image']).shape)
        print(batch['mask'].shape)
        print(batch['filter'].shape)
    # temp = nib.load(mask_dir + "/jiang qi_01^^^^ 004090_colon.nii")
    # print(temp.get_fdata().shape)
