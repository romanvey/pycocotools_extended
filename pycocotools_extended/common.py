import copy
import os
from collections import Counter, defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import random


def get_colors(num):
    return plt.cm.rainbow(np.linspace(0, 1, num))


def get_categories(data):
    return {cat_id: cat_data["name"] for cat_id, cat_data in data.cats.items()}


def get_meta_by_ann_id(data, ann_id, bboxes, categs):
    assert type(ann_id) is int
    anns = data.loadAnns(ann_id)
    return restructure_anns(anns, bboxes=bboxes, categs=categs)


def get_meta_by_img_id(data, img_id, bboxes, categs):
    assert type(img_id) is int
    anns = data.loadAnns(data.getAnnIds(imgIds=img_id))
    return restructure_anns(anns, bboxes=bboxes, categs=categs)


def restructure_anns(anns, bboxes, categs):
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


def train_test_split(anns_path, train_path='train.json', test_path='test.json', train_size=0.8, random_seed=0):
    json_data = json.load(open(anns_path))
    train_data, test_data = copy.deepcopy(json_data), copy.deepcopy(json_data)

    random.seed(random_seed)
    random.shuffle(test_data["images"])

    split_ind = int(len(test_data["images"]) * train_size)
    train_data["images"] = test_data["images"][:split_ind]
    test_data["images"] = test_data["images"][split_ind:]

    train_data["annotations"], test_data["annotations"] = [], []
    train_imgs = set(map(lambda x: x["id"], train_data["images"]))
    test_imgs = set(map(lambda x: x["id"], test_data["images"]))
    for ann in json_data["annotations"]:
        if ann["image_id"] in train_imgs:
            train_data["annotations"].append(ann)
        elif ann["image_id"] in test_imgs:
            test_data["annotations"].append(ann)

    with open(train_path, 'w') as f_train, open(test_path, 'w') as f_test:
        json.dump(train_data, f_train)
        json.dump(test_data, f_test)
