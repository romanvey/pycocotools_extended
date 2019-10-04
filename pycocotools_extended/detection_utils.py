import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import pycocotools_extended.common as common


def display_bboxes_by_img_id(data, img_id, imgs_path, transform=None, ax=None, fontsize=22, cat_names=None,
                             colors=None):
    assert type(img_id) is int
    out = common.get_meta_by_img_id(data, img_id, bboxes=True, categs=True)
    img = common.get_image_by_img_id(data, img_id, imgs_path)

    bboxes, categs = out['bboxes'], out['categs']
    if cat_names is None:
        cat_names = common.get_cat2name(data)
    if colors is None:
        colors = common.get_colors(len(cat_names))

    if transform is not None:
        out = transform(image=img, bboxes=bboxes, category_id=categs)
        img, bboxes, categs = out['image'], out['bboxes'], out['category_id']

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


def display_bboxes_by_img_ids(data, img_ids, imgs_path, transform=None, **kwargs):
    n_cols = 4
    n_rows = np.ceil(len(img_ids) / float(n_cols))
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))

    for i, img_id in enumerate(img_ids):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_title("Image %d" % img_id)
        display_bboxes_by_img_id(data, img_id, imgs_path=imgs_path, transform=transform, ax=ax, **kwargs)
    plt.show()


def filter_ann_ids_by_min_area(data, ann_ids, min_area=0):
    filtered_anns = []
    anns = data.loadAnns(ann_ids)
    for ann in anns:
        if ann['area'] < min_area:
            continue
        filtered_anns.append(ann['id'])
    return filtered_anns


def crop_image_by_bbox(img, bbox, padding=0):
    img_h, img_w = img.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(x + w + padding, img_w)
    y2 = min(y + h + padding, img_h)
    return img[y1:y2, x1:x2]


def get_cropped_bboxes_by_ann_ids(data, ann_ids, imgs_path, padding=0):
    cropped_imgs = []
    for ann_id in ann_ids:
        ann = data.loadAnns(ann_id)[0]
        img_id, bbox_loc, bbox_c = ann['image_id'], ann['bbox'], ann['category_id']
        img = common.get_image_by_img_id(data, img_id, imgs_path)
        cropped_img = crop_image_by_bbox(img, bbox_loc, padding)
        cropped_imgs.append(cropped_img)
    return cropped_imgs
