"""
    deep learning model for cell segmentation (Mask-RCNN)

    see *control panel* section of cellMaskRCNN.py for instructions
"""

# if __name__ == '__main__': # indent under when building documentation

# ----------------------------------------------------------------------------------------------------
# control panel

mode = 'deploy with PI2D'
# chose from options: 'train', 'test', 'deploy', 'deploy with PI2D';
# options 'deploy' and 'deploy with PI2D' have similar outputs,
# but the later is recommended for large images

num_epochs = 3
# number of train epochs

model_path = 'Models/cell-mask-rcnn-model.pt'
# where to save the model

dataset_path = 'DataForIS/TrainTest'
# path to folder containing images and labels;
# images should be 16-bit, single channel, .tif files;
# labels should be 8-bit .png files, where pixels from each cell have unique, identical, integer labels
# each corresponding pair (image, label) should have the same name (excluding the extension)

train_subset_fraction = 0.9
# fraction of dataset used to train; remaining goes to 'test' subset

deploy_path_in = 'DataForIS/DeployIn'
# folder containing images to perform inference on after training
# used under modes 'deploy' and 'deploy with PI2D';
# images should be 16-bit, single channel, .tif files;
# no other files should be in this folder

deploy_path_out = 'DataForIS/DeployOut'
# folder where outputs of inference are saved
# used under modes 'deploy' and 'deploy with PI2D'

suggested_patch_size = 400
# under mode 'deploy with PI2D', each image is square split in tiles
# where each side has approximately this many pixels in length

margin = 200
# under mode 'deploy with PI2D', each image is square split in tiles
# where adjacent tiles overlap by about this many pixels


# ----------------------------------------------------------------------------------------------------
# reference

# this script is adapted from
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://github.com/pytorch/tutorials/blob/master/_static/torchvision_finetuning_instance_segmentation.ipynb
#
# PyTorch tutorials are released under BSD 3-Clause License:
# https://github.com/pytorch/tutorials/blob/master/LICENSE;
# a copy of its text is appended at the end of this file


# ----------------------------------------------------------------------------------------------------
# machine room


import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
import torch.distributed as dist
import math
import matplotlib.pyplot as plt
import random

from gpfunctions import listfiles, tifread, imread, imshow, fileparts, imwrite, imerode, imgaussfilt, imadjust, imfillholes

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def uint16Gray_to_uint8RGB(I):
    assert I.dtype == 'uint16'
    I = imadjust(I.astype('float64')/65535)
    return np.uint8(255*np.stack([I,I,I],axis=2))

def uint16Gray_to_doubleGray(I):
    assert I.dtype == 'uint16'
    return imadjust(I.astype('float64')/65535)

def doubleGray_to_uint8RGB(I):
    assert I.dtype == 'float64'
    return np.uint8(255*np.stack([I,I,I],axis=2))

def get_labels(im, mk, bb, sc):
    pred_labels = np.zeros(im.shape)
    i_label = 0
    for i in range(bb.shape[0]):
        x0, y0, x1, y1 = np.round(bb[i,:]).astype(int)
        x1 = np.minimum(x1, im.shape[1]-1)
        y1 = np.minimum(y1, im.shape[0]-1)
        if sc[i] > 0.9:
            mask_box = np.zeros(im.shape, dtype=bool)
            mask_box[y0:y1,x0:x1] = True
            mask_i = np.logical_and(mk[i,:,:] > 0.5, mask_box)
            i_label += 1
            pred_labels[mask_i] = i_label

    return pred_labels

def get_boxes_and_contours(im, mk, bb, sc):
    boxes = []
    contours = []
    for i in range(bb.shape[0]):
        x0, y0, x1, y1 = np.round(bb[i,:]).astype(int)
        x1 = np.minimum(x1, im.shape[1]-1)
        y1 = np.minimum(y1, im.shape[0]-1)
        if sc[i] > 0.9:
            boxes.append([x0, y0, x1, y1])

            mask_box = np.zeros(im.shape, dtype=bool)
            mask_box[y0:y1,x0:x1] = True

            mask_i = np.logical_and(mk[i,:,:] > 0.5, mask_box)

            ct = np.logical_and(mask_i, np.logical_not(imerode(mask_i,1)))
            bd = np.zeros(mask_i.shape, dtype=bool)
            bd[0,:] = True; bd[-1,:] = True; bd[:,0] = True; bd[:,-1] = True
            bd = np.logical_and(bd, mask_i)
            ct = np.logical_or(ct, bd)

            ct_coords = np.argwhere(ct)
            contours.append(ct_coords)

    return boxes, contours

def draw_boxes_and_contours(im, boxes, contours):
    im2 = 0.9*np.copy(im)
    for idx in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[idx]
        ct = contours[idx]

        im2[ymin:ymax, xmin] = 1
        im2[ymin:ymax, xmax] = 1
        im2[ymin, xmin:xmax] = 1
        im2[ymax, xmin:xmax] = 1

        for idx_ct in range(ct.shape[0]):
            im2[ct[idx_ct,0], ct[idx_ct,1]] = 1

    return im2

class CellsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, load_annotations=True):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = listfiles(root, '.tif')
        self.ants = None
        if load_annotations:
            self.ants = listfiles(root, '.png')

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        img = uint16Gray_to_uint8RGB(tifread(img_path))
        
        target = None
        if self.ants:
            ant_path = self.ants[idx]
            mask = imread(ant_path)
        
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

if mode == 'train' or mode == 'test':
    # use our dataset and defined transformations
    dataset = CellsDataset(dataset_path, get_transform(train=True))
    dataset_test = CellsDataset(dataset_path, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    # print(indices)
    n_train = int(train_subset_fraction*len(dataset))
    dataset = torch.utils.data.Subset(dataset, indices[:n_train])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[n_train:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    print('n train', len(dataset), 'n test', len(dataset_test))

device_train = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device_train)

if mode == 'train':
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

def train_one_epoch(model, optimizer, data_loader, device):
    # model.train()
    model.to(device)

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
#         pdb.set_trace()

        # reduce losses over all GPUs for logging purposes
#         import pdb; pdb.set_trace()
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    return loss_value

# evaluate on gpu
@torch.no_grad()
def evaluate(model, data_loader, device):
    # model.eval()
    model.to(device)
    avg_dice_dataset = 0
    n_images = len(data_loader.dataset)
    rand_idx = np.random.randint(n_images)
    rand_log_img = None
    idx_image = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)#, targets)
        # when you call model(image,targets), it changes the size of ground truth masks in targets
        
        avg_dice_batch = 0
        for i in range(len(outputs)):
            out_i = outputs[i]
            ant_i = targets[i]
#             print(out_i.keys())
#             print(ant_i.keys())
            pred_mask, _ = torch.max(out_i['masks'],dim=0)
            pred_mask = torch.squeeze(pred_mask).cpu().numpy()
            grtr_mask, _ = torch.max(ant_i['masks'],dim=0)
            grtr_mask = grtr_mask.cpu().numpy().astype(np.float32)
            
            # import pdb; pdb.set_trace()
            # print(images[i].shape, pred_mask.shape, grtr_mask.shape)
            # pred_mask = imresizeDouble(pred_mask, list(grtr_mask.shape))
            
            # https://www.jeremyjordan.me/semantic-segmentation/
            # https://arxiv.org/pdf/1606.04797.pdf
            dice_coef = 2*np.sum(pred_mask*grtr_mask)/(np.sum(pred_mask**2)+np.sum(grtr_mask**2))
            avg_dice_dataset += dice_coef
            if idx_image == rand_idx:
                blue = np.zeros((pred_mask.shape))
                rand_log_img = np.stack([pred_mask, grtr_mask, blue],axis=2)
#                 imshow(images[i].cpu().numpy()[0,:,:])
#                 imshow(grtr_mask)
#                 print(images[i].cpu().numpy()[0,:,:].shape, grtr_mask.shape)
            idx_image += 1
    avg_dice_dataset /= n_images
#     print('avg_dice_dataset', avg_dice_dataset)
    return avg_dice_dataset, rand_log_img
            
#             print(pred_mask.shape, np.max(pred_mask), grtr_mask.shape, np.max(grtr_mask))
#             print(pred_mask.dtype, np.max(pred_mask), grtr_mask.dtype, np.max(grtr_mask))

# dice_coef, rand_img = evaluate(model, data_loader_test, device=device_train)
# print(dice_coef)
# imshow(rand_img)

if mode == 'train':
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        model.train()
        # import pdb; pdb.set_trace()
        loss_train = train_one_epoch(model, optimizer, data_loader, device_train)
        print('epoch', epoch, 'loss_train', loss_train)
        # update the learning rate
        lr_scheduler.step()
        
        # evaluate on the test dataset
        model.eval()
        dice_test, rand_img = evaluate(model, data_loader_test, device=device_train)
        print('epoch', epoch, 'dice_test', dice_test)
        # imshow(rand_img)

    torch.save(model.state_dict(), model_path)

if mode == 'test':
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        model.to(device_train)

        for img_index in range(len(dataset_test)):
            img, _ = dataset_test[img_index]
            prediction = model([img.to(device_train)])

            im = np.mean(img.numpy(),axis=0)

            p = prediction[0]['masks'][:, 0].cpu().numpy()
            p_max = np.max(p,axis=0)

            bb = prediction[0]['boxes'].cpu().numpy()
            sc = prediction[0]['scores'].cpu().numpy()

            im2 = 0.9*np.copy(im)
            fig = plt.figure(figsize=(12,6))
            for i in range(bb.shape[0]):
                x0, y0, x1, y1 = np.round(bb[i,:]).astype(int)
                x1 = np.minimum(x1, im2.shape[1]-1)
                y1 = np.minimum(y1, im2.shape[0]-1)
                if sc[i] > 0.9:
                    im2[y0:y1,x0] = 1
                    im2[y0:y1,x1] = 1
                    im2[y0,x0:x1] = 1
                    im2[y1,x0:x1] = 1
            plt.subplot(1,2,1)
            plt.imshow(im2,cmap='gray')
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(p_max)
            plt.axis('off')
            plt.show()

if mode == 'deploy':
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        model.to(device_train)

        dataset_deploy = CellsDataset(deploy_path_in, get_transform(train=False), False)

        for img_index in range(len(dataset_deploy)):
            file_path = dataset_deploy.imgs[img_index]
            _, file_name, _ = fileparts(file_path)
            print('processing image', file_name)

            img, _ = dataset_deploy[img_index]
            prediction = model([img.to(device_train)])

            im = np.mean(img.numpy(),axis=0)
            mk = prediction[0]['masks'][:, 0].cpu().numpy()
            bb = prediction[0]['boxes'].cpu().numpy()
            sc = prediction[0]['scores'].cpu().numpy()

            lb = get_labels(im, mk, bb, sc)
            # imwrite(np.uint8(255*im), deploy_path_out+'/'+file_name+'_input.png')
            imwrite(np.uint8(lb), deploy_path_out+'/'+file_name+'_prediction.png')

if mode == 'deploy with PI2D':
    from PartitionOfImageOM import PI2D

    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        model.to(device_train)

        im_path_list = listfiles(deploy_path_in, '.tif')

        for img_index in range(len(im_path_list)):
            file_path = im_path_list[img_index]
            _, file_name, _ = fileparts(file_path)
            print('processing image', file_name)

            img_tif = tifread(file_path)


            # prediction without PI2D

            img = torch.tensor(np.transpose(uint16Gray_to_uint8RGB(img_tif), [2, 0, 1]).astype(np.float32)/255)

            prediction = model([img.to(device_train)])

            im = np.mean(img.numpy(),axis=0)
            mk = prediction[0]['masks'][:, 0].cpu().numpy()
            bb = prediction[0]['boxes'].cpu().numpy()
            sc = prediction[0]['scores'].cpu().numpy()

            # boxes, contours = get_boxes_and_contours(im, mk, bb, sc)
            # im2 = draw_boxes_and_contours(im, boxes, contours)

            lb = get_labels(im, mk, bb, sc)
            imwrite(np.uint8(lb), deploy_path_out+'/'+file_name+'_1_full_size_prediction.png')


            # prediction with PI2D

            img_double = uint16Gray_to_doubleGray(img_tif)

            PI2D.setup(img_double,suggested_patch_size,margin)

            for i_patch in range(PI2D.NumPatches):
                P = PI2D.getPatch(i_patch)
                P3 = doubleGray_to_uint8RGB(P)

                img = torch.tensor(np.transpose(P3, [2, 0, 1]).astype(np.float32)/255)

                prediction = model([img.to(device_train)])

                im = np.mean(img.numpy(),axis=0)
                mk = prediction[0]['masks'][:, 0].cpu().numpy()
                bb = prediction[0]['boxes'].cpu().numpy()
                sc = prediction[0]['scores'].cpu().numpy()

                boxes, contours = get_boxes_and_contours(im, mk, bb, sc)

                PI2D.patchOutput(i_patch, boxes, contours)
                
            contours = PI2D.Contours
            lb = np.zeros(img_double.shape)
            for idx in range(len(contours)):
                ct = contours[idx]
                mk = np.zeros(lb.shape, dtype=bool)
                for rc in ct:
                    mk[rc[0],rc[1]] = True
                mk = imfillholes(mk)
                lb[mk] = idx+1

            imwrite(np.uint8(lb), deploy_path_out+'/'+file_name+'_2_PI2D_prediction.png')





# ----------------------------------------------------------------------------------------------------

# BSD 3-Clause License

# Copyright (c) 2017, Pytorch contributors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.