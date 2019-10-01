import albumentations as albu


def create_transform(aug_fn, size=None, normalize=True, bboxes=True, min_visibility=0., mean=None, std=None,
                     bbox_params=None):
    pipeline = []
    process_fn = None

    if size is not None:
        if type(size) is int:
            resize_fn = albu.Resize(size, size)
        else:
            resize_fn = albu.Resize(*size)
        aug_fn.insert(0, resize_fn)

    if bboxes:
        if bbox_params is None:
            bbox_params = {
                'format': 'coco',
                'min_visibility': min_visibility,
                'label_fields': ['category_id']
            }
    else:
        bbox_params = None
    aug_fn = albu.Compose(aug_fn, bbox_params=bbox_params)
    pipeline.append(aug_fn)

    if normalize:
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        normalize_fn = albu.Normalize(mean=mean, std=std)
        pipeline.append(normalize_fn)

    pipeline = albu.Compose(pipeline)

    def process(image, bboxes, category_id):
        r = pipeline(image=image, bboxes=bboxes, category_id=category_id)
        return r['image'], r['bboxes'], r['category_id']

    def process_without_bboxes(image):
        r = pipeline(image=image)
        return r['image']

    if bboxes:
        process_fn = process
    else:
        process_fn = process_without_bboxes
    return process_fn