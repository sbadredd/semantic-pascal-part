import numpy as np
import torch
import detectron2
import detectron2.model_zoo
import detectron2.engine
import detectron2.config
import detectron2.structures

import utils.structures

def get_predictor(
        use_cuda: bool,
        checkpoint_url: str = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    ) -> detectron2.engine.DefaultPredictor:
    """
    Args:
        use_cuda (bool): if False, use cpu.
        checkpoint_url (str, optional): URL of a trained model checkpoint.
            Defaults to "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml".
            See https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py 

    Returns:
        detectron2.engine.DefaultPredictor: detectron2 predictor
    """
    cfg = detectron2.config.get_cfg()
    if not use_cuda:
        cfg.MODEL.DEVICE = "cpu"
    checkpoint_url = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(detectron2.model_zoo.get_config_file(checkpoint_url))
    cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(checkpoint_url)
    return detectron2.engine.DefaultPredictor(cfg)

def boxes_to_detectron2(
        boxes: list[utils.structures.Box],
        to_cuda: bool
) -> detectron2.structures.Boxes:
    boxes_np = np.stack([b.numpy() for b in boxes], axis=0)
    boxes_tensor = torch.from_numpy(boxes_np)
    if to_cuda:
        boxes_tensor = boxes_tensor.cuda()
    return detectron2.structures.Boxes(boxes_tensor)

def im_to_tensor(
        im_array: np.ndarray,
        to_cuda: bool
) -> torch.Tensor:
    """
    Args:
        im_array (np.ndarray): Shape (H,W,C)

    Returns:
        torch.Tensor: Shape (C,H,W)
    """
    im_tensor = torch.tensor(im_array)
    im_tensor = torch.permute(im_tensor, [2,0,1])
    im_tensor = im_tensor.float()
    if to_cuda:
        im_tensor = im_tensor.cuda()
    return im_tensor

def im_tensors_to_detectron2(
        im_tensors: list[torch.Tensor],
        predictor: detectron2.engine.DefaultPredictor
) -> detectron2.structures.ImageList:
    """Converts a list of N image tensor of (C,H,W) shape to H', W' must be a multiple of 
    `predictor.size_divisibility`.
    """
    return detectron2.structures.ImageList.from_tensors(
            im_tensors, predictor.model.backbone.size_divisibility)

def scale_image_and_boxes(
        raw_image: torch.Tensor,
        new_image: torch.Tensor,
        raw_boxes: detectron2.structures.Boxes
) -> detectron2.structures.Boxes:
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[-2:]
        new_height, new_width = new_image.shape[-2:]
        new_boxes = raw_boxes.clone()
        new_boxes.scale(scale_x=new_width/raw_width, scale_y=new_height/raw_height)
    return new_boxes

def run_backbone_and_roi_head_with_given_boxes(
        predictor: detectron2.engine.DefaultPredictor,
        x: torch.Tensor,
        box_lists: list[detectron2.structures.Boxes]
) -> torch.Tensor:
    """Call detectron2 backbone + ROIPooler + box head (transform and flatten the region features)
    given a list of boxes in an image.
    Doesn't call the box predictor (FastRCNNOutputLayers).

    Args:
        predictor (detectron2.engine.DefaultPredictor): detectron2 predictor.
        x (torch.Tensor): A batch of N images in a tensor of (N,C,H,W) shape.
                H, W must be a multiple of `predictor.size_divisibility`.
        box_lists (list[Boxes]): A list of N Boxes or N RotatedBoxes, where N is the number of 
                images in the batch. The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

    Returns:
        torch.Tensor: A tensor of shape (M, output_size) where M is the total number of
                boxes aggregated over all N batch images.
    """
    with torch.no_grad():
        features = predictor.model.backbone(x)
        features_list = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads.box_pooler(features_list, box_lists)
        box_features = predictor.model.roi_heads.box_head(box_features)
    return box_features