import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def get_colors(num):
    return plt.cm.rainbow(np.linspace(0, 1, num))


def get_categories(data):
    return {cat_id: cat_data["name"] for cat_id, cat_data in data.cats.items()}


def get_image_by_img_id(data, img_id, imgs_root):
    assert type(img_id) is int
    img_path = os.path.join(imgs_root, data.loadImgs(img_id)[0]["file_name"])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return img


def get_meta_by_img_id(data, img_id):
    assert type(img_id) is int
    anns = data.loadAnns(data.getAnnIds(imgIds=img_id))
    bboxes, categs = [], []
    for ann in anns:
        bboxes.append(ann["bbox"])
        categs.append(ann["category_id"])
    return bboxes, categs


def get_image_by_ann_id(data, ann_id):
    assert type(ann_id) is int
    pass


def get_meta_by_ann_id(data, ann_id):
    assert type(ann_id) is int
    pass


def display_bboxes_by_img_id(data, img_id, imgs_root, transform=None, ax=None, fontsize=22):
    assert type(img_id) is int
    bboxes, categs = get_meta_by_img_id(data, img_id)
    img = get_image_by_img_id(data, img_id, imgs_root)
    cat_names = get_categories(data)
    colors = get_colors(len(cat_names))

    if transform is not None:
        img, bboxes, categs = transform(img, bboxes, categs)

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(30, 20))
    ax.imshow(img)
    for bbox_loc, bbox_c in zip(bboxes, categs):
        rect = patches.Rectangle((bbox_loc[0], bbox_loc[1]), bbox_loc[2], bbox_loc[3], linewidth=3,
                                 edgecolor=colors[bbox_c - 1], facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox_loc[0], bbox_loc[1] - 4, cat_names[bbox_c], fontsize=fontsize, color=colors[bbox_c - 1])
    if ax is None:
        plt.show()


def display_bboxes_by_img_ids(data, img_ids, imgs_root, transform=None):
    n_cols = 4
    n_rows = np.ceil(len(img_ids) / float(n_cols))
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))

    for i, img_id in enumerate(img_ids):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_title("Image %d" % img_id)
        display_bboxes_by_img_id(data, img_id, imgs_root=imgs_root, transform=transform, ax=ax)
    plt.show()
