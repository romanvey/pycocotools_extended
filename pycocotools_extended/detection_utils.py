from .common import get_meta_by_img_id, get_image_by_img_id, get_colors, get_categories
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def display_bboxes_by_img_id(data, img_id, imgs_path, transform=None, ax=None, fontsize=22):
    assert type(img_id) is int
    out = get_meta_by_img_id(data, img_id)
    bboxes, categs = out['bboxes'], out['categs']
    img = get_image_by_img_id(data, img_id, imgs_path)
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
