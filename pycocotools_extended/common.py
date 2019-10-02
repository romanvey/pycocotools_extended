import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import copy


def get_colors(num):
    return plt.cm.rainbow(np.linspace(0, 1, num))


def get_categories(data):
    return {cat_id: cat_data["name"] for cat_id, cat_data in data.cats.items()}


def get_meta_by_ann_id(data, ann_id, bboxes=True, categs=True):
    assert type(ann_id) is int
    anns = data.loadAnns(ann_id)
    return restructure_anns(anns, bboxes=bboxes, categs=categs)


def get_meta_by_img_id(data, img_id, bboxes=True, categs=True):
    assert type(img_id) is int
    anns = data.loadAnns(data.getAnnIds(imgIds=img_id))
    return restructure_anns(anns, bboxes=bboxes, categs=categs)


def restructure_anns(anns, bboxes=True, categs=True):
    out = defaultdict(list)
    for ann in anns:
        if bboxes:
            out['bboxes'].append(ann['bbox'])
        if categs:
            out['categs'].append(ann['category_id'])
    return out


def get_image_by_ann_id(data, ann_id, imgs_path):
    assert type(ann_id) is int
    img_id = data.getImgIds(annIds=ann_id)[0]
    return get_image_by_img_id(img_id, imgs_path)


def get_image_by_img_id(data, img_id, imgs_path):
    assert type(img_id) is int
    img_path = os.path.join(imgs_path, data.loadImgs(img_id)[0]["file_name"])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return img


def calculate_categories(data):
    ann_ids = data.getAnnIds()
    anns = data.loadAnns(ann_ids)
    c = Counter()
    for ann in anns:
        c[ann['category_id']] += 1
    return c


def _crop_img(img, bbox, padding=0):
    img_h, img_w = img.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(x + w + padding, img_w)
    y2 = min(y + h + padding, img_h)
    return img[y1:y2, x1:x2]


def get_cropped_bboxes_by_ann_ids(data, ann_ids, padding=0):
    cropped = []
    for ann_id in ann_ids:
        ann = data.loadAnns(ann_id)[0]
        img_id = ann['image_id']
        bbox_c = ann['category_id']
        bbox_loc = ann['bbox']
        bbox_area = ann['area']
        img = get_image_by_img_id(img_id)
        cropped.append(_crop_img(img, bbox_loc, padding))
    return cropped


def merge_datasets(*datasets):
    data, img_prefix = datasets[0]
    data = copy.deepcopy(data)

    # first dataset image filename mapping
    for img in data["images"]:
        img["file_name"] = os.path.join(img_prefix, img["file_name"])

    if len(datasets) == 1:
        return data

    # other datasets mappings
    img_id = max(map(lambda x: x["id"], data["images"])) + 1
    ann_id = max(map(lambda x: x["id"], data["annotations"])) + 1
    for dataset, img_prefix in datasets[1:]:
        # image id and filename mapping
        img_id_mapping = {}
        for img in dataset["images"]:
            img_id_mapping[img["id"]] = img_id
            img["id"] = img_id
            img["file_name"] = os.path.join(img_prefix, img["file_name"])
            data["images"].append(img)
            img_id += 1

        # annotations image_id and id mapping
        for ann in dataset["annotations"]:
            ann["image_id"] = img_id_mapping[ann["image_id"]]
            ann["id"] = ann_id
            data["annotations"].append(ann)
            ann_id += 1
    return data
