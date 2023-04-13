import tqdm

import h5py

import utils.postprocess.structures

def save_box_data(
        file_path: str, 
        box_datas: list[utils.postprocess.structures.BoxData]
):
    with h5py.File(file_path, "w") as f:
        for box_data in tqdm.tqdm(box_datas, desc="Saving box features in hdf5"):
            box_group = f.create_group(str(box_data.id_))
            box_group.create_dataset("roi_features", data=box_data.roi_features)
            box_group.create_dataset("position", data=box_data.position)
            box_group.attrs["type"] = box_data.type_
            box_group.attrs["partof"] = box_data.partof_id
            box_group.attrs["pic"] = box_data.pic

def save_paired_data_as_dict_of_dict(
        file_path: str,
        paired_datas: list[utils.postprocess.structures.PairedData]
):
    with h5py.File(file_path, "w") as f:
        for paired_data in tqdm.tqdm(paired_datas, desc="Saving pairs and partof labels in hdf5"):
            try:
                x1 = f[str(paired_data.id1)]
            except KeyError:
                x1 = f.create_group(str(paired_data.id1))
            x1x2 = x1.create_group(str(paired_data.id2))
            x1x2.attrs["ispartof"] = paired_data.ispartof

def read_box_from_hdf5(box_data_file: h5py.File, box_id: int):
    return utils.postprocess.structures.BoxData(
            id_=box_id,
            roi_features=box_data_file[box_id]["roi_features"][()],
            position=box_data_file[box_id]["position"][()],
            partof_id=box_data_file[box_id].attrs["partof"],
            type_=box_data_file[box_id].attrs["type"],
            pic=box_data_file[box_id].attrs["pic"]
    )
