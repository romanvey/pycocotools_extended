import torch
import numpy as np
import os
import random

import pycocotools_extended.coco_ext


class COCOdetection:
    def __init__(self, anns_path, imgs_path, transform=None):
        self.coco_ext = pycocotools_extended.coco_ext.COCOext(anns_path=anns_path, imgs_path=imgs_path)
        self.img_ids = list(self.coco_ext().getImgIds())
        self.transform = transform

    def __getitem__(self, idx):
        try:
            img, bboxes, categs = self.get_data(idx)
            height, width, _ = img.shape
            img = self._preprocess_img(img)
            target = self._preprocess_target(bboxes, categs, height, width)
        except ValueError as e:
            print(e)
            return self[random.randrange(len(self))]
        return img, target

    def __len__(self):
        return len(self.img_ids)

    def get_data(self, idx):
        img_id = self.img_ids[idx]
        img = self.coco_ext.get_image_by_img_id(img_id)
        out = self.coco_ext.get_meta_by_img_id(img_id, bboxes=True, categs=True)
        bboxes, categs = out['bboxes'], out['categs']
        if self.transform is not None:
            out = self.transform(image=img, bboxes=bboxes, category_id=categs)
            img, bboxes, categs = out['image'], out['bboxes'], out['category_id']
        return img, bboxes, categs

    def get_img_path(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.coco_ext().loadImgs(img_id)[0]['file_name']
        return os.path.join(self.coco_ext.imgs_path, img_path)

    @staticmethod
    def _preprocess_img(img):
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

    @staticmethod
    def _preprocess_target(bboxes, labels, height, width):
        bboxes = np.array(bboxes)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        bboxes = bboxes / np.array([width, height, width, height])
        target = np.hstack([bboxes, np.expand_dims(labels, axis=1)])
        target = target.astype(np.float32)
        return torch.from_numpy(target)
