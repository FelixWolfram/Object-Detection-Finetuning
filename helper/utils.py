import os
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import ResNet
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision.ops import nms
from torchvision import tv_tensors
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.ops import generalized_box_iou_loss
from torch import nn
import numpy as np
import wandb
import re
from copy import deepcopy


WEIGHTS = ResNet50_Weights.DEFAULT # weights = IMAGENET1K_V2
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# -------------- Classes --------------

class CustomImageDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None, img_transform=None):
        self.annotations_dir = annotations_dir
        self.annot_file_names = os.listdir(self.annotations_dir)
        self.img_dir = img_dir
        self.transform = transform  # v2 transformations which will be applied to image + label
        self.img_transform = img_transform # normal transformations (which will be also applied to validation + test) like resize, crop and normalization

    def __len__(self):
        return len(self.annot_file_names)

    def __getitem__(self, idx):
        file_name = self.annot_file_names[idx]
        # build the path to the annotation.txt and read the content
        annotation_path = os.path.join(self.annotations_dir, file_name)
        with open(annotation_path, "r") as f:
            try:
                objects = f.read().strip().split("\n")
                yolo_boxes = [[float(x.strip()) for x in obj.strip().split(" ")] for obj in objects]  # transform the string to a readable float label array
                # annotations as lines in txt-file with no header, each line containes 5 values seperated by a space: (class_id, x_center, y_center, width, height)
            except Exception as e:
                # no object/label for this image => file is empty, so use an empty label-list as the GT-annotation
                yolo_boxes = []
        # build the related path to the image and load it as a tensor
        img_path = os.path.join(self.img_dir, file_name).removesuffix("txt") + "jpg"
        image = decode_image(img_path)
        current_img_size = image.shape[-1] # image is a tensor with shape [C, H, W]
        # apply transformations if necessary
        if self.transform:
            if len(yolo_boxes) > 0: # only do the bounding box transformations if there are bounding boxes in the image
                raw_boxes = np.array([[cx, cy, w, h] for _, cx, cy, w, h in yolo_boxes]) 
                cxcywh_boxes = raw_boxes * current_img_size # cxcywh boxes is noted with absolute pixels, yolo with relative from [0, 1], so multiply by image size
                # => boxes are relative to the original image size, so we multiply by the current image size (instead of the target image size) to get the current absolute pixel values for the bounding box
                bboxes = tv_tensors.BoundingBoxes(cxcywh_boxes, format="CXCYWH", canvas_size=(current_img_size, current_img_size)) # canvas_size is the size of the original image
                image, transformed_bboxes = self.transform(image, bboxes) # transform both the image and the bounding boxes (only with v2 transformations which also affect the bounding boxes)
                relative_bboxes = transformed_bboxes / image.shape[-1] # scale back to relative bounding boxes ([0, 1]) for the yolo format => now relative to the new image size

                # to convert back to the yolo label format, the class labels have to be added again
                classes = torch.tensor([lbl[0] for lbl in yolo_boxes]).unsqueeze(1) # unsqueeze(1) to get a shape of [N, 1] => bboxes are shape [N, 4] so we can concatenate them to get a shape of [N, 5] for the yolo_boxes
                yolo_boxes = torch.cat((classes, relative_bboxes), dim=1) # dim=1 to concatenate along the columns 
            else:
                image = self.transform(image) # if there are no bounding boxes, just transform the image
        if self.img_transform:
            # preprocess transformations like normalize, which will not affect the label
            image = self.img_transform(image)
        return image, yolo_boxes


class ObjectDetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(2048)
        self.conv2 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(2048)
        self.outputmapping = nn.Conv2d(in_channels=2048, out_channels=20, kernel_size=1, bias=True)


    def forward(self, x):
        # extra Layer 1
        x = F.relu(self.batchnorm1(self.conv1(x)))
        # extra Layer 2
        x = F.relu(self.batchnorm2(self.conv2(x)))
        # mapping on desired output Tensor shape
        x = self.outputmapping(x)
        return x


# -------------- Model --------------

def load_initial_model_and_transform(frozen: bool = True) -> tuple[ResNet, v2.Compose, int]:
    # load the preprocessing transforms from the weights
    preprocess = WEIGHTS.transforms()
    # extract elements from preprocess to use them with v2 (so labels and images can be transformed together)
    mean = preprocess.mean
    std = preprocess.std
    # double the input size (=> doubles output size)
    resize_size = preprocess.resize_size[0] * 2
    crop_size = preprocess.crop_size[0] * 2

    model_transform = v2.Compose([
        v2.Resize(resize_size),
        v2.CenterCrop(crop_size),
        v2.Normalize(mean=mean, std=std)
    ])

    # load model, either frozen or unfrozen 
    model = resnet50(weights=WEIGHTS) 
    if frozen:
        for param in model.parameters():
            param.requires_grad = False # freeze the parameters of the model to only train the new layers for object detection first

    return model, model_transform, crop_size


def load_ObjDet_model(frozen: bool = True):
    """
    Loads the initial model and swaps the classification head with the object detection head.
    Returns the modified model, the preprocessing transforms for the ResNet50 model and the image size after the preprocessing transformations.
    """
    model, test_transform, image_size = load_initial_model_and_transform(frozen=frozen)
    # get all the elements of the network as a list => the last two layers (avg_pool and fc) are then stripped including the forward pass of those modules
    model = nn.Sequential(*list(model.children())[:-2])
    # add the object detection head to the model
    model.add_module("Object Detection Head", ObjectDetectionHead())
    return model, test_transform, image_size


# -------------- Dataset and Dataloader --------------

def load_default_train_transforms(image_size: int, test_transform: v2.Compose) -> v2.Compose:
    """
    Loads the default training transformations for the object detection model. 
    Includes random resized crop, random perspective, color jitter and random grayscale.
    """
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(image_size, (0.6, 1)),
        v2.RandomPerspective(0.75, 0.2),
        v2.ColorJitter(0.3, 0.2, 0.2, 0.1),
        v2.RandomGrayscale(0.1)
    ])

    return train_transforms


def load_default_datasets(train_transform: v2.Compose, test_transform: v2.Compose, dataset_path: str) -> tuple[CustomImageDataset, CustomImageDataset, CustomImageDataset]:
    train_dataset =  CustomImageDataset(
        annotations_dir=f"{dataset_path}/train/labels",
        img_dir=f"{dataset_path}/train/images",
        transform=train_transform,
    )

    val_dataset =  CustomImageDataset(
        annotations_dir=f"{dataset_path}/valid/labels",
        img_dir=f"{dataset_path}/valid/images",
        img_transform=test_transform
    )

    test_dataset =  CustomImageDataset(
        annotations_dir=f"{dataset_path}/test/labels",
        img_dir=f"{dataset_path}/test/images",
        img_transform=test_transform
    )

    return train_dataset, val_dataset, test_dataset


def collate_fn(batch):
    return list(zip(*batch))
    # with this we prevent an error when returning the batch in the dataloader
    # Pytorch wants to order the batch like this: [(image_1, target_1), (image_2, target_2), ...]
    # here the shape of all images and of all targets must be the same, but with object detection, the shape of the target object varies
    # so we sort new with this by zipping all images and all targets together and turning back into a list:  [(image_1, image_2, ...), (target_1, target_2, ...)] 
    #       => the size of the first and the second tuple are now both 64 


def load_default_dataloaders(batch_size: int, num_workers: int, image_size: int, test_transform: v2.Compose, train_transform: v2.Compose = None, dataset_path: str = "kaggle/Traffic_Sign/car") -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the Custom Traffic Sign Detection Dataset and wraps it in DataLoaders for training, validation and testing.
    By standard it also applies the custom transformations for data augmentation during training. The test_transform is the preprocessing transformation for the ResNet50 model (build when loading the model).
    """
    if train_transform is None:
        # using standard training transformations if no custom ones are provided
        train_transform = load_default_train_transforms(image_size, test_transform)

    train_dataset, val_dataset, test_dataset = load_default_datasets(train_transform, test_transform, dataset_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


# -------------- Loss Function --------------

def get_target_maps(map_size: int, batch_size: int, y: list, device: str = DEVICE):
    """
    Transforming the yolo format labels in the format (torch.tensor((N, 4)), ..., torch.tensor((N, 4))) where N represents the number of objects in the image into three target maps used for calculating the loss.
    **The three target maps are:**<br>
    1. Objectness map - binary, 1 if an object is present in the cell, 0 otherwise
    2. Class map - integer, the class of the object in the cell, -1 if no object is present
    3. Location map - tensor of shape (4, map_size, map_size), containing the relative coordinates and dimensions of the object in the cell
    """
    # to get the correct 2D Feature map for the objectness "map" for each batch item, get the x_center and y_center (relative to image size) of all objects, multiple by image size (14) and store in a tensor
    obj_map = torch.zeros(batch_size, map_size, map_size).to(device)
    class_maps = torch.ones(batch_size, map_size, map_size, dtype=torch.long).to(device) * -1
    loc_maps = torch.zeros(batch_size, 4, map_size, map_size).to(device)

    for b_idx, b_item in enumerate(y):
        for object in b_item:
            (c, x_cent, y_cent, w, h) = object
            x_coord, y_coord = min(13, int(x_cent * map_size)), min(13, int(y_cent * map_size)) # min(13, ...) because when x_cent or y_cent is 1.0 an OutOfBoundsError would occur (index = 14)
            x_offset, y_offset = (x_cent * map_size - x_coord), (y_cent * map_size - y_coord)

            obj_map[b_idx, y_coord, x_coord] = 1
            class_maps[b_idx, y_coord, x_coord] = int(c)
            loc_maps[b_idx, :, y_coord, x_coord] = torch.tensor([x_offset, y_offset, w, h]) # inject the id tensor along the second axis by getting all element of that pixel for alle 4 feature maps
        
    return obj_map, class_maps, loc_maps


def yolo_to_bbox(yolo_box: torch.Tensor, img_size: int, map_size: int, device: str = DEVICE) -> torch.Tensor:
    """Conver the yolo format boxes (x_offset, y_offset, width, height) where x_offset and y_offset are relative to the grid cell into the bounding box format (x1, y1, x2, y2) needed for calculating the GIoU loss."""
    # yolo box shape: (64, 4, 14, 14)
    cell_width = img_size / map_size

    # yolo_box[:, 0:2] is now only the x-offset and the y-offset feature map, but for GIoU we need the absolute coordinates of x1 and y1
    x_scale_tensor = torch.arange(0, map_size).expand(map_size, -1).to(device) # arange creates vector [0, ..., map_size-1], expand duplicates that to reach shape (map_size x map_size)
    y_scale_tensor = torch.arange(0, map_size).view((-1, 1)).expand(-1, map_size).to(device) # here we need to change the row vector to a "column vector", then expand "to the right"
    # calculate absolute x-coordinates => every column needs to be multiplied by 32 times the column index
    abs_x = cell_width * x_scale_tensor + yolo_box[:, 0] * cell_width 
    # calculate absolute y-coordinates => same as x-coordinate calculation but with 32 times the row index
    abs_y = cell_width * y_scale_tensor + yolo_box[:, 1] * cell_width
    
    # also convert relative height/width into absolute height/width
    abs_w = yolo_box[:, 2] * img_size
    abs_h = yolo_box[:, 3] * img_size

    x1 = abs_x - abs_w / 2
    y1 = abs_y - abs_h / 2
    x2 = abs_x + abs_w / 2
    y2 = abs_y + abs_h / 2

    bbox_tensor = torch.stack((x1, y1, x2, y2), dim=3)
    return bbox_tensor


def objectness_loss(pred: torch.Tensor, target_obj: torch.Tensor, no_obj_w: float) -> torch.Tensor:
    """
    Calculates the objectness loss for predicting whether an object is present in a grid cell by calculating Sigmoid and Binary Cross Entropy Loss for each pixel, as multiple objects can be present in an image.
    """
    # using Sigmoid + BCE for the objectness score as there can be mulitple objects in an image (higher weight on objectness = 1 because there are more pixels with no objects than with objects)
    pred_obj = pred[:, 0] 
    loss_obj = F.binary_cross_entropy_with_logits(input=pred_obj, target=target_obj.float(), reduction="none") # this function automatically applies sigmoid function to the logits
    # as there are way more pixels with no objects, weight no object pixels less
    weight_map = torch.where(target_obj == 1, torch.ones_like(loss_obj), torch.ones_like(loss_obj) * no_obj_w) 
    loss_obj *= weight_map # multiply by no_obj_w, where no object should be detected - inplace with masked_fill_
    return loss_obj.mean() # mean over all pixels and batches


def localization_loss(pred: torch.Tensor, target_loc: torch.Tensor, img_size: int, map_size: int, target_obj: torch.Tensor, loc_l2_factor: float) -> torch.Tensor:
    """
    Calculates the localization loss for bounding box coordinates by applying GIoU and L2 distance penalty for width and height.

    **Reason for additional L2 distnace penalty:**<br>
    The additional distance penalty is necessary, because otherwise the model does not correctly localize small images. When only using the GIoU-Loss, the gradient is really small with a small IoU,
    especially if the GT bounding box is localized inside the predicted bounding box, so the ladder penalty term of the GIoU has no effect.
    A larger prior layer size additionaly increases this problem as gradients are unstable when breaking out of that plateu. Also with small boxes the GIoU-Loss can be too harsh,
    L2 helps weighting those imbalances and shrinking sizing the predicted bounding box to the correct size, where the GIoU-Loss is then more effective.
    """
    # using GIoU + L2 distance penalty for width and height => GIoU to normalize for different sizes of objects
    # by only using GIoU tho, the model can't confidently localize small objects, which is adressed by the L2 distance penalty => encourages for smaller boxes, helping the model get out of the small gradients plateau at the beginning of training 
    yolo_pred = torch.sigmoid(pred[:, 1:5])
    bbox_pred = yolo_to_bbox(yolo_pred, img_size=img_size, map_size=map_size)
    bbox_target = yolo_to_bbox(target_loc, img_size=img_size, map_size=map_size)
    # prediction and target localization tensors have shape [64, 14, 14, 4], but for GIoU loss, shape [64, 14, 14, 4] is needed

    # calculated GIoU loss and widht and height L2 distance penalty
    all_giou_loss = generalized_box_iou_loss(bbox_pred, bbox_target)
    l2_w = loc_l2_factor * (yolo_pred[:, 2].sqrt() - target_loc[:, 2].sqrt()) ** 2
    l2_h = loc_l2_factor * (yolo_pred[:, 3].sqrt() - target_loc[:, 3].sqrt()) ** 2

    # weight them accordingly and only calculate the loss for pixels where an object is present (target_obj = 1)
    giou_l2_loss = all_giou_loss + l2_w + l2_h
    mean_relevant_giou_loss = torch.masked_select(giou_l2_loss, target_obj > 0).mean()

    return mean_relevant_giou_loss


def classification_loss(pred: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
    """
    Calculates the classification loss by applying softmax and cross entropy loss on the feature maps used for predicting the class of the object. 
    """
    # Softmax + Cross Entropy for the classification loss (only one object per cell can be detected, so only one class is possible) => loss is only calculated on cells with objectness = 1
    class_logits = pred[:, 5:]
    loss_class = F.cross_entropy(class_logits, target_class, reduction="mean", ignore_index=-1) # ignore all indices with no class assignnment (objectness = 0) => class = -1
    # Cross Entropy loss automaticall calculates the softmax over dim =1 (softmax for each pixel between all feature maps) and compares with the correct class index with target_value = 1 for that class
    # with standard reduction = "mean" the mean is taken over the loss for every pixel over every batch so Sum_of_BCE / (64 * 14 * 14) => the 25 dimensions from the classes are reduced to one dimension (CE loss)
    return loss_class


def calc_detection_loss(
        pred: torch.Tensor, 
        y: tuple[torch.Tensor, ...], 
        img_size: int, 
        device: str = DEVICE, 
        no_obj_w: float = 0.2,
        loc_w: float = 4,
        obj_w: float = 12,
        class_w: float = 1,
        loc_l2_factor: float = 1
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates all three losses for a prediction and weights them to account for the different orders of magnitude and importances.
    """
    map_size = pred.shape[-1]

    target_obj, target_class, target_loc = get_target_maps(map_size=map_size, batch_size=pred.shape[0], y=y, device=device)
    
    loss_obj = objectness_loss(pred, target_obj, no_obj_w=no_obj_w)
    loss_class = classification_loss(pred, target_class)
    loss_loc = localization_loss(pred, target_loc, img_size=img_size, map_size=map_size, target_obj=target_obj, l2_factor=loc_l2_factor)


    loss = obj_w * loss_obj + class_w * loss_class + loc_w * loss_loc
    return loss, obj_w * loss_obj, class_w * loss_class, loc_w * loss_loc


# -------------- Training and Testing Loop --------------

def train_loop(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: str = DEVICE) -> list[tuple[float, float, float, float]]:
    """Training loop for each epoch. Iterates over all batches in the dataloader, calculates the loss, backpropagates and updates the model parameters while saving the loss for each batch."""
    model.train()

    batch_losses = []
    all_batch_losses = []

    for batch in dataloader:
        X, y = torch.stack(batch[0]).to(device), batch[1]

        pred = model(X)
        loss, loss_obj, loss_class, loss_loc = calc_detection_loss(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        batch_losses.append(loss.item())
        all_batch_losses.append((loss.item(), loss_obj.item(), loss_class.item(), loss_loc.item()))

    return all_batch_losses

        
def test_loop(model: nn.Module, dataloader: DataLoader, device: str = DEVICE) -> list[tuple[float, float, float, float]]:
    """Testing loop for each epoch. Iterates over all batches in the dataloader and calculates the loss for each batch."""
    model.eval() 

    batch_losses = []
    all_batch_losses = []

    with torch.no_grad():
        for batch in dataloader:
            X, y = torch.stack(batch[0]).to(device), batch[1]

            pred = model(X)
            loss, loss_obj, loss_class, loss_loc = calc_detection_loss(pred, y)
            batch_losses.append(loss.item())

            all_batch_losses.append((loss.item(), loss_obj.item(), loss_class.item(), loss_loc.item()))

    return all_batch_losses


def log_metrics(run, epoch, mean_train_losses, mean_val_losses, total_mean_train_batch_loss, total_mean_val_batch_loss, labels):
    """
    Logs the training and validation losses to Weights & Biases (wandb) and to the command line.
    """
    print(f"-------------------------------\nEpoch {epoch+1}")
    print("Average Train Loss:", total_mean_train_batch_loss)
    print("Average Validation Loss:", total_mean_val_batch_loss)

    # because wandb.log() can only log scalars, we convert the list into a dict which will be then logged
    log_dict = {"epoch": epoch}
    for label, train_l, val_l in zip(labels, mean_train_losses, mean_val_losses):
        # create key-value pairs for each metric for the dict
        log_dict[f"train/{label}"] = train_l
        log_dict[f"val/{label}"] = val_l
    run.log(log_dict)


def train_model(model, train_dataloader, val_dataloader, optimizer, wandb_project_name, wandb_config, epochs):
    """
    Trains the model for a given number of epochs, logging the training and validation losses to Weights & Biases (wandb) as well as in the command line.
    The losses are all collected and the best model (with the lowest validation loss) is saved.
    """
    # load environment variables from dotenv to get the wandb API key
    train_losses = []
    val_losses = []
    best_model_dict = None
    best_val_loss = float("inf")

    labels = ["Total_Loss", "Objectness_Loss", "Classification_Loss", "Localization_Loss"]

    with wandb.init(project=wandb_project_name, config=wandb_config) as run:
        run.watch(model)

        for epoch in range(epochs):
            train_batch_losses = train_loop(model, train_dataloader, optimizer)
            val_batch_losses = test_loop(model, val_dataloader)

            mean_train_losses = [np.array(l).mean() for l in zip(*train_batch_losses)]
            train_losses.append(mean_train_losses)
            total_mean_train_batch_loss = mean_train_losses[0]

            mean_val_losses = [np.array(l).mean() for l in zip(*val_batch_losses)]
            val_losses.append(mean_val_losses)
            total_mean_val_batch_loss = mean_val_losses[0]

            if total_mean_val_batch_loss < best_val_loss:
                best_val_loss = total_mean_val_batch_loss
                best_model_dict = deepcopy(model.state_dict())
                run.summary["best_val_loss"] = best_val_loss    # save the validition loss so it can be seen in the dashboard
            
            log_metrics(
                run,
                epoch,
                mean_train_losses,
                mean_val_losses,
                total_mean_train_batch_loss,
                total_mean_val_batch_loss,
                labels
            )

    torch.save(best_model_dict, f"best_models/TrafficSign_ObjDet_{best_val_loss:.4f}.pth")

    return train_losses, val_losses, best_val_loss


# -------------- Visualization --------------

def get_box_coords(box, image_size):
    class_id, x_c, y_c, w, h = box
    width = w * image_size
    height = h * image_size
    x1 = x_c * image_size - width / 2
    y1 = y_c * image_size - height / 2
    return (x1, y1, width, height)


def get_coords_from_pred(pred, threshold, image_size):
    class_maps = torch.softmax(pred[:, 5:], dim=1)
    max_probs, class_ids = torch.max(class_maps, dim=1)

    obj_map = torch.sigmoid(pred[:, 0])
    cell_confs = obj_map * max_probs # to get the total confidence for each cell, multiply the cell confidence with the class probability
    obj_mask = cell_confs > threshold
    obj_indices = torch.nonzero(obj_mask, as_tuple=True)
    # obj_indices list of length 3, each with 64 elements
    # first element are the batch indices, second element the y-coordinates (row) and third element the x-coordinates (column) of cells with objectness > 0.2

    obj_classes = class_ids[obj_indices] # because obj_indices is a tuple (batch_idx_tensor, y_coord_tensor, x_coord_tensor) we can index like class_ids[batch_idx_tensor, y_coord_tensor, x_coord_tensor]
    obj_conf = cell_confs[obj_indices] 

    loc_maps = torch.sigmoid(pred[:, 1:5])
    bbox_preds = yolo_to_bbox(loc_maps, img_size=image_size, map_size=pred.shape[-1])
    obj_bboxes = bbox_preds[obj_indices]

    return torch.stack(obj_indices, dim=1), obj_classes, obj_conf, obj_bboxes


def get_bbox_preds(batch_size, pred, image_size, threshold=0.25, nms_treshold=0.5):
    """
    Gets the predicted bounding boxes, classes and confidence scores for each image in the batch after applying non-maximum suppression (NMS.
    The bounding boxes are returned in the format (x1, y1, x2, y2).
    """
    obj_indices, classes, confidences, bboxes = get_coords_from_pred(pred, threshold=threshold, image_size=image_size)

    nms_classes = []
    nms_bboxes = []
    nms_confidences = []

    # as each image only has one eye and therefore one prediction, we just iterate over the first predictions/images
    for i in range(batch_size):
        # draw the boxes around the correct objects
        pred_mask = obj_indices[:, 0] == i
        obj_conf = confidences[pred_mask] # confidence scores of all predictions above the threshold for the current image
        obj_bboxes = bboxes[pred_mask]  # bbox predictions for object predictions above the threshold for the current image

        nms_indices = nms(obj_bboxes, obj_conf, iou_threshold=nms_treshold) # return the indices of the bbox-tensor (pred_loc) that are kept after nms
        nms_preds_indices = torch.nonzero(pred_mask)[nms_indices] # because the indices are relative to the pred_loc tensor, we calculate the absolute indices of the predictions (to index the ouput of get_coords_from_pred())

        nms_pred_classes = classes[nms_preds_indices] # all relevant classes after nms
        nms_pred_bboxes = obj_bboxes[nms_indices] # filter the bbox predictions to only keep the ones that are kept after nms
        nms_pred_conf = obj_conf[nms_indices]

        nms_classes.append(nms_pred_classes)
        nms_bboxes.append(nms_pred_bboxes)
        nms_confidences.append(nms_pred_conf)

    return nms_classes, nms_bboxes, nms_confidences


def visualize_bbox_preds(X, pred_classes, pred_bboxes, confidences, image_size, test_transforms, class_names, y=None):
    """
    Visualize the predicted bounding boxes in the image in comparison to the ground truth bounding boxes.
    The first 16 images are shown in a 4x4 grid with the predicted bounding boxes in red and the ground truth bounding boxes in green.
    """
    fig, axs = plt.subplots(4, 4, figsize=(15, 12))
    axs = axs.flatten()

    normalize = test_transforms[-1]
    mean = torch.tensor(normalize.mean).view(3, 1, 1) # because we have a tensor with shape [3, 244, 244], to process each channel seperately, we have to create a tensor with shape [3, 1, 1]
    std  = torch.tensor(normalize.std).view(3, 1, 1)

    # as each image only has one eye and therefore one prediction, we just iterate over the first predictions/images
    for i in range(16):
        # show the image with the classified label
        # X[i] is the tensor of the normalized image (with mean and std defined above) => with those parameters, we convert back to the normal image
        normal_image = X[i] * std + mean # revert the normalization process by multiplying with the standard deviation and adding the mean
        
        ax = axs[i]
        ax.imshow(normal_image.permute(1, 2, 0))
        ax.axis("off")
        
        # iterate over every prediction for the current image, get the coordinates and draw them into the current image
        for (cls, loc, conf) in zip(pred_classes[i], pred_bboxes[i], confidences[i]):
            x1, y1, x2, y2 = loc.cpu().detach().numpy()
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            pred_class_idx = cls.item()
            ax.annotate(f"{class_names[pred_class_idx]} - {conf.item() * 100:.2f}%", 
                        xy=(x1, y1 - image_size * 0.02), 
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'), 
                        fontsize=8)
        # correct prediction in green (if available)
        if y:
            x1_gt, y1_gt, w_gt, h_gt = get_box_coords(y[i][0], image_size)
            ax.add_patch(plt.Rectangle((x1_gt, y1_gt), w_gt, h_gt, fill=False, edgecolor='green', linewidth=2))

        # => torchvision.util.draw_bounding_boxes() can also draw bounding boxes in a given image

    plt.tight_layout()
    plt.show()


# -------------- Inference --------------

def load_best_model(model, file_name_start="TrafficSign_ObjDet_", folder="best_models"):
    # extracting the loss on the validation dataset with a regex expression, to be able identify the best model, which will be loaded
    saved_models = [model_dict for model_dict in os.listdir(folder) if model_dict.startswith(file_name_start)]
    # search for float in the name, get only the match-string and convert to float to be able to sort (+ argmax) afterwards to get the best model index in saved_models
    best_loss_idx = np.sort(np.array([float(re.search("[+-]?([0-9]*[.])?[0-9]+", model).group()) for model in saved_models])).argmax() 
    best_model_file = saved_models[best_loss_idx]
    print("Best model file:", best_model_file)
        
    best_model_dict = torch.load(f"{folder}/{best_model_file}", map_location=DEVICE)
    model.load_state_dict(best_model_dict)