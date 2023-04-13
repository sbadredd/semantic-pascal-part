import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import collections
import h5py
import cv2

import utils.postprocess.structures
import utils.postprocess.hdf5


box_h5_file = h5py.File(os.path.join("output","processed","test","box_features.hdf5"))
image_folder = os.path.join("data","JPEGImages_test")
box_ids = [17736,17737]
box_ids = [str(x) for x in box_ids]

boxes = [utils.postprocess.hdf5.read_box_from_hdf5(box_h5_file, id_) for id_ in box_ids]
pic_to_boxes: dict[int,list[utils.postprocess.structures.BoxData]] = collections.defaultdict(list)
for box in boxes:
    pic_to_boxes[box.pic].append(box)

for pic, boxes in pic_to_boxes.items():
    pic_str = str(pic)
    pic_str = "0"*(6-len(pic_str)) + pic_str + ".jpg"
    im = cv2.imread(os.path.join(image_folder, pic_str))
    colors = [(36,255,12),(255,12,36)]
    for box,color in zip(boxes,colors):
        x1, y1, x2, y2 = box.position[0], box.position[1], box.position[2], box.position[3]
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        cv2.putText(im, box.type_, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow(pic_str,im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()