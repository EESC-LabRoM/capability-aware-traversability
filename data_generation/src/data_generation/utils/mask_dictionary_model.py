# Adapted from Grounded-SAM-2 (https://github.com/IDEA-Research/Grounded-SAM-2), utils/mask_dictionary_model.py
# Copyright 2023 - present, IDEA Research.
# Licensed under the Apache License, Version 2.0, see data_generation/licenses/Grounded-SAM-2-LICENSE
# Modified for CAT (type hints, docstrings, optional output directories, removed debug code).
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Any

@dataclass
class MaskDictionaryModel:
    """
    Instance annotations of one frame.

    :ivar mask_name: str - mask file name
    :ivar mask_height: int - mask height in pixels
    :ivar mask_width: int - mask width in pixels
    :ivar promote_type: str - promoted annotation type
    :ivar labels: dict - instance id to ObjectInfo
    """
    mask_name:str = ""
    mask_height: int = 1080
    mask_width:int = 1920
    promote_type:str = "mask"
    labels:dict = field(default_factory=dict)

    def add_new_frame_annotation(self, mask_list, box_list, label_list, background_value = 0) -> None:
        """
        Replace the annotation with per-instance masks, boxes and labels; ids are `background_value + index + 1`.

        :param mask_list: torch.Tensor | np.ndarray - (N, H, W) bool masks
        :param box_list: list - (x1, y1, x2, y2) boxes, one per mask
        :param label_list: list[str] - class names, one per mask
        :param background_value: int - offset added to every instance id
        """
        mask_img = torch.zeros(mask_list.shape[-2:])
        anno_2d = {}
        for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
            final_index = background_value + idx + 1

            if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            mask_img[mask == True] = final_index
            name = label
            box = box
            new_annotation = ObjectInfo(instance_id = final_index, mask = mask, class_name = name, x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3])
            anno_2d[final_index] = new_annotation

        self.mask_height = mask_img.shape[0]
        self.mask_width = mask_img.shape[1]
        self.labels = anno_2d

    def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0) -> int:
        """
        Relabel instances with the id of the first tracked object whose IoU exceeds the threshold, else a new id; empty masks are dropped.

        :param tracking_annotation_dict: MaskDictionaryModel - tracked annotation to match
        :param iou_threshold: float - minimum mask IoU for a match
        :param objects_count: int - instance ids assigned so far
        :return: int - updated count of assigned instance ids
        """
        updated_masks = {}

        for seg_obj_id, seg_mask in self.labels.items():
            flag = 0
            new_mask_copy = ObjectInfo()
            if seg_mask.mask.sum() == 0:
                continue

            for object_id, object_info in tracking_annotation_dict.labels.items():
                iou = self.calculate_iou(seg_mask.mask, object_info.mask)
                if iou > iou_threshold:
                    flag = object_info.instance_id
                    new_mask_copy.mask = seg_mask.mask
                    new_mask_copy.instance_id = object_info.instance_id
                    new_mask_copy.class_name = seg_mask.class_name
                    break

            if not flag:
                objects_count += 1
                flag = objects_count
                new_mask_copy.instance_id = objects_count
                new_mask_copy.mask = seg_mask.mask
                new_mask_copy.class_name = seg_mask.class_name
            updated_masks[flag] = new_mask_copy
        self.labels = updated_masks
        return objects_count

    def get_target_class_name(self, instance_id) -> str:
        """
        Return an instance's class name.

        :param instance_id: int - key in `labels`
        :return: str - class name
        """
        return self.labels[instance_id].class_name


    @staticmethod
    def calculate_iou(mask1, mask2) -> torch.Tensor:
        """
        Calculate the IoU of two masks.

        :param mask1: torch.Tensor - first mask
        :param mask2: torch.Tensor - second mask of the same shape
        :return: torch.Tensor - scalar IoU, NaN if both masks are empty
        """
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)

        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection

        iou = intersection / union
        return iou

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dict.

        :return: dict[str, Any] - mask name, size, promote type and labels as dicts
        """
        return {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }

@dataclass
class ObjectInfo:
    """
    Annotation of one object instance.

    :ivar instance_id: int - instance id
    :ivar mask: torch.Tensor | None - instance mask
    :ivar class_name: str - class name
    :ivar x1: int - left box coordinate
    :ivar y1: int - top box coordinate
    :ivar x2: int - right box coordinate
    :ivar y2: int - bottom box coordinate
    :ivar logit: float - detection logit
    """
    instance_id:int = 0
    mask: any = None
    class_name:str = ""
    x1:int = 0
    y1:int = 0
    x2:int = 0
    y2:int = 0
    logit:float = 0.0

    def update_box(self) -> list[int] | None:
        """
        Set the box to the extent of nonzero mask pixels.

        :return: list[int] | None - empty list if the mask has no nonzero pixels, otherwise None
        """
        nonzero_indices = torch.nonzero(self.mask)

        if nonzero_indices.size(0) == 0:
            return []

        y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
        y_max, x_max = torch.max(nonzero_indices, dim=0)[0]

        bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dict, excluding the mask.

        :return: dict[str, Any] - instance id, class name, box coordinates and logit
        """
        return {
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "logit": self.logit
        }