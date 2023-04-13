import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import utils.postprocess.csv
import utils.postprocess.structures
import utils.postprocess.hdf5


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, 
            help="Which data to process. Options: 'trainval' or 'test'.")
args_dict = vars(parser.parse_args())

data_type = args_dict["data"]


output_dir = os.path.join("output","raw",data_type)
classes_file_path = os.path.join("data","classes.csv")
box_datas = utils.postprocess.csv.read_csv_output(output_dir, classes_file_path)
paired_datas = utils.postprocess.structures.get_paired_data_of_bbs_in_same_pictures(box_datas)

utils.postprocess.hdf5.save_box_data(
    file_path=os.path.join("output","processed",data_type,"box_features.hdf5"), 
    box_datas=box_datas)
utils.postprocess.hdf5.save_paired_data_as_dict_of_dict(
    file_path=os.path.join("output","processed",data_type,"pairs_partof.hdf5"), 
    paired_datas=paired_datas)