import os
import random

import pandas as pd
import numpy as np

import colorsys
import cv2
import albumentations as A
from tqdm.auto import tqdm

import torch

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def decode_rle_mask(rle_mask, shape=(520, 704)):
    """
    Decode run-length encoded segmentation mask string into 2d array

    Parameters
    ----------
    rle_mask (str): Run-length encoded segmentation mask string
    shape (tuple): Height and width of the mask

    Returns
    -------
    mask [numpy.ndarray of shape (height, width)]: Decoded 2d segmentation mask
    """

    rle_mask = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle_mask[0:][::2], rle_mask[1:][::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros((shape[0] * shape[1]), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    mask = mask.reshape(shape[0], shape[1])
    mask = np.uint8(mask)
    return mask


def encode_rle_mask(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 1
    rle[1::2] -= rle[::2]
    return rle.tolist()


def get_bboxes_from_mask(masks):
    coco_boxes = []
    for mask in masks:
        pos = np.nonzero(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        coco_boxes.append([xmin, ymin, xmax, ymax])
    coco_boxes = np.asarray(coco_boxes)
    return coco_boxes


def get_targets_mask(df, img_id):
    targets = df[df['id'] == img_id]['cell_type'].apply(lambda x: cls_map[x]).values
    rles = df[df['id'] == img_id]['annotation'].values
    return targets, rles


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def plot_image_annotations(image, masks, bboxes, labels, aug=None):
    image = image.copy()

    colors = {unique_lbl: random_colors(1, True)[0] for unique_lbl in np.unique(labels)}

    if aug is not None:
        augmented = aug(image=image, masks=masks, bboxes=bboxes,
                        labels=labels)
        image = augmented['image']
        masks = augmented['masks']
        bboxes = augmented['bboxes']

    bboxes = np.stack(bboxes).astype(int)

    for idx, box in enumerate(bboxes):
        color = tuple([int(value * 255) for value in colors[labels[idx]]])
        image = cv2.rectangle(image, (box[2], box[3]), (box[0], box[1]), color=color, thickness=2)

    for idx, mask in enumerate(masks):
        color = colors[labels[idx]]
        image = apply_mask(image, mask, color)

    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.show()


class CellSegData(torch.utils.data.Dataset):
    def __init__(self, root, df, split='train', aug=None, cls_map=None):
        self.augmentations = aug
        self.cls_map = cls_map

        train, test = train_test_split(df['id'].unique(), train_size=0.9)
        if split == 'train':
            self.dataset = train
        else:
            self.dataset = test

        self.dict_df = {img_id: df[df['id'] == img_id] for img_id in tqdm(self.dataset)}
        self.root = root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_id = self.dataset[index]
        image = cv2.imread(os.path.join(self.root, img_id + '.png'))

        info = self.dict_df[img_id]
        n_objects = len(info['annotation'])

        labels = info['cell_type'].apply(lambda x: self.cls_map[x]).values
        rles = info['annotation'].values

        masks = []
        for mask in rles:
            decoded_mask = decode_rle_mask(rle_mask=mask, shape=image.shape)
            masks.append(decoded_mask)

        bboxes = get_bboxes_from_mask(masks)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image, masks=masks, bboxes=bboxes,
                                           labels=labels)
            image = augmented['image']
            masks = augmented['masks']
            bboxes = augmented['bboxes']
            bboxes = np.stack(bboxes).astype(int)

        masks = np.asarray(masks)

        bboxes = torch.as_tensor(bboxes, dtype=torch.int64)

        is_bad_labels = False
        degenerate_boxes = bboxes[:, 2:] <= bboxes[:, :2]
        if degenerate_boxes.any():
            is_bad_labels = True
            # print the first degenerate box
            bb_idxs = torch.where(degenerate_boxes.any(dim=1))[0].numpy()

        labels = torch.as_tensor([1 for i in range(len(bboxes))], dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        if is_bad_labels:
            bboxes = bboxes[[i for i in range(len(bboxes)) if i not in bb_idxs]]
            labels = labels[[i for i in range(len(bboxes)) if i not in bb_idxs]]
            masks = masks[[i for i in range(len(bboxes)) if i not in bb_idxs]]

        image_id = torch.tensor([index])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        # This is the required target for the Mask R-CNN
        target = {
            'boxes': bboxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        image = image.transpose((2, 0, 1))
        return torch.Tensor(image), target


def analyze_train_sample(model, ds_train, sample_index):
    img, targets = ds_train[sample_index]
    plt.imshow(img.numpy().astype(np.uint8).transpose((1, 2, 0)))
    plt.title("Image")
    plt.show()

    masks = np.zeros((640, 640))
    for mask in targets['masks']:
        masks = np.logical_or(masks, mask)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.imshow(masks, alpha=0.3)
    plt.title("Ground truth")
    plt.show()

    model.eval()
    with torch.no_grad():
        preds = model([img.cuda()])[0]

    plt.imshow(img.cpu().numpy().transpose((1, 2, 0)))
    all_preds_masks = np.zeros((640, 640))
    for mask in preds['masks'].cpu().detach().numpy():
        all_preds_masks = np.logical_or(all_preds_masks, mask[0] > 0.5)
    plt.imshow(all_preds_masks, alpha=0.4)
    plt.title("Predictions")
    plt.show()


if __name__ == '__main__':

    train_df = pd.read_csv('data/train.csv')

    # In[7]:

    cls_map = {value: idx for idx, value in enumerate(train_df['cell_type'].unique())}

    train_augmentations = A.Compose([
        A.Resize(640, 640),
        # A.RandomResizedCrop(640, 640, scale=(0.8, 1.0), ratio=(0.9, 1.3)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90()
    ], bbox_params={"format": "pascal_voc", "min_area": 0, "min_visibility": 0, 'label_fields': ['labels']})

    # let's train it for 10 epochs
    num_epochs = 200

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    # use our dataset and defined transformations
    dataset = CellSegData('data/train', train_df, 'train', train_augmentations, cls_map)
    dataset_test = CellSegData('data/train', train_df, 'test', train_augmentations, cls_map)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=2, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    NUM_CLASSES = 2

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,
                                                               box_detections_per_img=600)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    # weights = torch.load('weights/checkpoint.pth')
    # move model to the right device
    # model.state_dict(weights['model'])
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001)
    # optimizer.load_state_dict(weights['optimizer'])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    output_dir = 'weights'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs)):
        # train for one epoch, printing every 10 iterations
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        for images, targets in metric_logger.log_every(data_loader, 10, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        analyze_train_sample(model, dataset, 20)
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)
        if output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            utils.save_on_master(checkpoint, os.path.join(output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(output_dir, "checkpoint.pth"))
