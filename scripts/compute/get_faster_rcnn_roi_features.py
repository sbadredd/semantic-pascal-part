import argparse
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import glob
import tqdm

import utils.annotation
import utils.detectron2
import utils.structures
import utils.saving

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, 
            help="Which data to process. Options: 'trainval' or 'test'.")
args_dict = vars(parser.parse_args())

data_type = args_dict["data"]

# Data loading
annot_dir = os.path.join("data", f"Annotations_{data_type}")
im_dir = os.path.join("data", f"JPEGImages_{data_type}")

annot_file_paths = glob.glob(os.path.join(annot_dir, "*.xml"))
annot_file_paths = sorted(annot_file_paths)
annots_lists = [utils.annotation.read_objects_in_annotation_file(f) for f in annot_file_paths]
annots_im_infos = [utils.annotation.read_image_info_in_annotation_file(f) for f in annot_file_paths]

# Preparing data saving
classes_to_id = utils.saving.get_classes_to_id(os.path.join("data","classes.csv"))
box_id_offset = 0
features_file_path = os.path.join("output","raw",data_type,"features.csv")
partof_file_path = os.path.join("output","raw",data_type,"partOf.csv")
types_file_path = os.path.join("output","raw",data_type,"types.csv")
if os.path.exists(features_file_path):
    os.remove(features_file_path)
if os.path.exists(partof_file_path):
    os.remove(partof_file_path)
if os.path.exists(types_file_path):
    os.remove(types_file_path)
    
# Feature extraction
predictor = utils.detectron2.get_predictor(use_cuda=False)
for i in tqdm.tqdm(range(len(annots_lists)), desc="Processing files..."):
    annots = annots_lists[i]
    im_info = annots_im_infos[i]

    im_array = im_info.read_image(im_dir)
    im_tensor = utils.detectron2.im_to_tensor(im_array, to_cuda=False)
    im_list_d2 = utils.detectron2.im_tensors_to_detectron2([im_tensor], predictor)
    new_im_tensor = im_list_d2.tensor[0]

    boxes = [annot.box for annot in annots]
    unscaled_boxes_d2 = utils.detectron2.boxes_to_detectron2(boxes, to_cuda=False)
    boxes_d2 = utils.detectron2.scale_image_and_boxes(im_tensor, new_im_tensor, unscaled_boxes_d2)
    boxes_list_d2 = [boxes_d2]

    features_tensor = utils.detectron2.run_backbone_and_roi_head_with_given_boxes(
            predictor, im_list_d2.tensor, boxes_list_d2
    )
    utils.saving.save_boxes_features(
        features=features_tensor, annots=annots, im_info=im_info, box_id_offset=box_id_offset,
        classes_to_id=classes_to_id, features_file_path=features_file_path, 
        partof_file_path=partof_file_path, types_file_path=types_file_path
    )
    box_id_offset += len(annots)