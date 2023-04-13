import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import glob
import tqdm

import cv2

import utils.annotation
import utils.box

def crop_bounding_boxes(
        annotation_file_path: str,
        source_images_dir: str,
        target_images_dir: str,
        square_the_box: bool = False,
        size_threshold: int = 6
):
    objects = utils.annotation.read_objects_in_annotation_file(annotation_file_path, size_threshold=size_threshold)
    im_info = utils.annotation.read_image_info_in_annotation_file(annotation_file_path)
    boxes = [obj.box for obj in objects]
    if square_the_box:
        boxes = [utils.box.adjust_to_square(box, im_info.width, im_info.height) for box in boxes]

    im = cv2.imread(os.path.join(source_images_dir, im_info.file))
    for (obj, box) in zip(objects, boxes):
        crop_im = im[box.ymin:box.ymax, box.xmin:box.xmax]
        object_filename = im_info.file[:-len(".jpg")] + f"_{obj.id}.jpg"
        cv2.imwrite(os.path.join(target_images_dir, object_filename), crop_im)

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, 
            help="Which data to process. Options: 'trainval' or 'test'.")
args_dict = vars(parser.parse_args())
data_type = args_dict["data"]

annotation_dir = os.path.join("data", f"Annotations_{data_type}")
source_images_dir = os.path.join("data", f"JPEGImages_{data_type}")
target_images_dir = os.path.join("data", f"JPEG_boxes_{data_type}")
xml_file_paths = glob.glob(os.path.join(annotation_dir, "*.xml"))
for filepath in tqdm.tqdm(xml_file_paths, desc="Processing images..."):
    crop_bounding_boxes(filepath, source_images_dir, target_images_dir)
