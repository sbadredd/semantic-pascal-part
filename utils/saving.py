import os
import csv

import torch

import utils.structures

def save_boxes_features(
        features: torch.Tensor,
        annots: list[utils.structures.ObjectAnnotation], 
        im_info: utils.structures.ImageInfo,
        box_id_offset: int,
        classes_to_id: dict[str, int],
        features_file_path: str,
        partof_file_path: str,
        types_file_path: str
    ) -> None:
    with open(features_file_path, "a+") as f:
        for (i, annot) in enumerate(annots):
            line_values = []
            line_values.append(im_info.file_basename)
            line_values.extend(features[i].numpy().tolist())
            box = annot.box
            line_values.extend([box.xmin, box.ymin, box.xmax, box.ymax])
            f.write(",".join(map(str,line_values)))
            f.write("\n")
    with open(partof_file_path, "a+") as f:
        for annot in annots:
            ispartof = box_id_offset+annot.ispartof if annot.ispartof != -1 else annot.ispartof
            f.write(str(ispartof))
            f.write("\n")
    with open(types_file_path, "a+") as f:
        for annot in annots:
            f.write(str(get_type_id(annot, classes_to_id)))
            f.write("\n")

def get_classes_to_id(classes_csv_file_path) -> dict[str,int]:
    """Read classes->id from the disk.

    Returns:
        dict[str,int]: key = class label, value = class id
    """
    with open(os.path.join(classes_csv_file_path)) as f:
            csv_reader = csv.reader(f)
            classes_to_id = {}
            id_ = 0
            for row in csv_reader:
                classes_to_id[row[0]] = id_
                id_ += 1
    return classes_to_id

def get_type_id(annot: utils.structures.ObjectAnnotation, classes_to_id: dict[str, int]) -> int:
    try:
        id_ = classes_to_id[annot.name]
    except KeyError as e:
        try:
            id_ = classes_to_id[annot.name.lower()]
        except:
            raise e
    return id_