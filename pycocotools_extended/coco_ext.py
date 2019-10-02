from pycocotools.coco import COCO

import pycocotools_extended.common as common
import pycocotools_extended.detection_utils as du


class COCOext:
    def __init__(self, anns_path, imgs_path):
        self.coco = COCO(annotation_file=anns_path)
        self.anns_path = anns_path
        self.imgs_path = imgs_path

        self.cat_names = common.get_categories(self.coco)
        self.colors = common.get_colors(len(self.cat_names))

    def __call__(self):
        return self.coco

    def get_meta_by_ann_id(self, ann_id, bboxes, categs):
        return common.get_meta_by_ann_id(self.coco, ann_id, bboxes, categs)

    def get_meta_by_img_id(self, img_id, bboxes, categs):
        return common.get_meta_by_img_id(self.coco, img_id, bboxes, categs)

    def get_image_by_ann_id(self, ann_id):
        return common.get_image_by_ann_id(self.coco, ann_id, self.imgs_path)

    def get_image_by_img_id(self, img_id):
        return common.get_image_by_img_id(self.coco, img_id, self.imgs_path)

    def calculate_categories(self):
        return common.calculate_categories(self.coco)

    def get_cropped_bboxes_by_ann_ids(self, ann_ids, padding=0):
        return du.get_cropped_bboxes_by_ann_ids(self.coco, ann_ids, self.imgs_path, padding=padding)

    def display_bboxes_by_img_id(self, img_id, transform=None, **kwargs):
        return du.display_bboxes_by_img_id(self.coco, img_id, self.imgs_path, transform=transform, colors=self.colors,
                                           cat_names=self.cat_names, **kwargs)

    def display_bboxes_by_img_ids(self, img_ids, transform=None, **kwargs):
        return du.display_bboxes_by_img_ids(self.coco, img_ids, self.imgs_path, transform=transform, colors=self.colors,
                                            cat_names=self.cat_names, **kwargs)

    def filter_ann_ids_by_min_area(self, ann_ids, min_area=0):
        return du.filter_ann_ids_by_min_area(self.coco, ann_ids, min_area)

    def train_test_split(self, train_path='train.json', test_path='test.json', train_size=0.8, random_seed=0):
        return common.train_test_split(self.anns_path, train_path, test_path, train_size, random_seed)
