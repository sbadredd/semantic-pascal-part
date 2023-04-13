import collections
from dataclasses import dataclass
import tqdm
import numpy as np

@dataclass
class BoxData:
    """
    Output data of a box in an image.

    Attributes:
        id (int): ID of the box in the whole dataset.
        roi_features (np.ndarray): Output of an object detector
        position (np.ndarray): Array of shape [4]. Containsm, in order: xmin, ymin, xmax, ymax.
        type_ (str): Class of the object.
        partof_id (int): ID of a container object in which this object belongs. If such container
                object does not exist, the value is -1.
        pic (int): ID of a picture in which the object belongs.
    """
    id_: int
    roi_features: np.ndarray
    position: np.ndarray
    type_: str
    partof_id: int
    pic: int

    def __post_init__(self):
        self.id_ = int(self.id_)
        self.roi_features = self.roi_features.astype(np.float32)
        self.position = self.position.astype(np.int32)
        self.type_ = str(self.type_)
        self.partof_id = int(self.partof_id)
        self.pic = int(self.pic) 

@dataclass
class PairedData:
    id1: int
    id2: int
    ispartof: bool


def get_paired_data_of_bbs_in_same_pictures(box_datas: list[BoxData]) -> list[PairedData]:
    """Grouping the data of pairs of bounding boxes that appear in the same picture.

    Args:
        box_datas (list[BoxData]): _description_

    Returns:
        list[PairedData]: _description_
    """
    pic_to_bbs = collections.defaultdict(set)
    pairs_where_1in2 = set()
    for box_data in tqdm.tqdm(box_datas, desc="Loading pair labels..."):
        pic_to_bbs[box_data.pic].add(box_data.id_)
        if box_data.partof_id != -1:
            pairs_where_1in2.add((box_data.id_, box_data.partof_id))
    
    paired_datas = []
    for pic in tqdm.tqdm(pic_to_bbs.keys(), desc="Saving pair labels in Python structures"):
        for i in pic_to_bbs[pic]:
            for j in pic_to_bbs[pic]:
                paired_datas.append(
                        PairedData(id1=i, id2=j, ispartof=((i,j) in pairs_where_1in2)))
    return paired_datas