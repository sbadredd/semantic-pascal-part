import os
import glob
import xml.etree.ElementTree as ET
import argparse

def indent_xml_files(annotation_dir: str):
    xml_filepaths = glob.glob(os.path.join(annotation_dir,"*.xml"))
    for filepath in xml_filepaths:
        tree = ET.parse(filepath)
        ET.indent(tree, '  ')
        tree.write(filepath, encoding="utf-8", xml_declaration=True)


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, 
            help="Which data to process. Options: 'trainval' or 'test'.")
args_dict = vars(parser.parse_args())
data_type = args_dict["data"]

dir = os.path.join("pascal_dataset","semantic_pascal_part",f"Annotations_{data_type}")
indent_xml_files(dir)
