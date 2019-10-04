import albumentations as albu
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from pycocotools_extended.transforms import create_transform
import pycocotools_extended.common as pe_common
import pycocotools_extended.detection_utils as du

ann_path = '/home/roman/Workspace/datasets/rect_dataset/full.json'
imgs_path = '/home/roman/Workspace/datasets/rect_dataset'

augs = {
    'rotate': [albu.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT, p=1.)]
}
aug_fn = augs['rotate']

coco = COCO(ann_path)
transform = create_transform(aug_fn, size=256, normalize=False)

du.display_bboxes_by_img_ids(coco, list(range(20)), imgs_path)
du.display_bboxes_by_img_ids(coco, list(range(20)), imgs_path, transform=transform)


img_id = 0

img = pe_common.get_image_by_img_id(coco, img_id, imgs_path)
bboxes, categs = pe_common.get_meta_by_img_id(coco, img_id)
du.display_bboxes_by_img_id(coco, img_id, imgs_path, transform)

img, bboxes, categs = transform(img, bboxes, categs)

img = pe_common.get_image_by_img_id(coco, img_id, imgs_path)
transform_without_bboxes = create_transform(aug_fn, normalize=False, bboxes=False)
img = pe_common.get_image_by_img_id(coco, img_id, imgs_path)
img = transform(img)
plt.imshow(img)
