import os
import tqdm
import csv
import numpy as np
import utils.postprocess.structures

def get_id_to_classes(classes_file_path: str) -> dict[int,str]:
    """Read id->classes from the disk.

    Returns:
        dict[int,str]: key = class id, value = class label
    """
    with open(os.path.join(classes_file_path)) as f:
        csv_reader = csv.reader(f)
        id_to_classes = {}
        id_ = 0
        for row in csv_reader:
            id_to_classes[id_] = row[0]
            id_ += 1
    return id_to_classes

def read_csv_output(
    data_dir: str,
    classes_file_path: str
)-> list[utils.postprocess.structures.BoxData]:
    """Fetch output data.
    TODO: if your system lacks system memory, consider reading the output in batches and 
    converting it to hdf5.

    Args:
        data_dir (str): Directory of the trainval or test data.

    Returns:
        list[BoxData]: _description_
    """
    id_to_class = get_id_to_classes(classes_file_path)
    print("Loading box features...")
    features_data = np.genfromtxt(os.path.join(data_dir, "features.csv"), delimiter=",", 
            dtype=np.float32)
    pics_data = features_data[:,0].astype(np.int32)
    pos_data = features_data[:,-4:]
    roi_features_data = features_data[:,1:-4]
    type_data = np.genfromtxt(os.path.join(data_dir, "types.csv"), dtype=np.int32)
    partOf_data = np.genfromtxt(os.path.join(data_dir, "partOf.csv"), dtype=np.int32)
    
    box_datas = []
    for i in tqdm.tqdm(range(len(features_data)), desc="Saving features in Python structures"):
        box_datas.append(
                utils.postprocess.structures.BoxData(id_=i,
                        roi_features=roi_features_data[i],
                        position=pos_data[i],
                        type_=id_to_class[type_data[i]],
                        partof_id=partOf_data[i],
                        pic=pics_data[i])
        )
    return box_datas
