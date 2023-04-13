import xml.etree.ElementTree as ET

import utils.structures

def read_objects_in_annotation_file(
        filepath: str,
        size_threshold: int = 6
    ) -> list[utils.structures.ObjectAnnotation]:
    annotations = []
    tree = ET.parse(filepath)
    image_file = tree.getroot().find("filename").text
    old_id_to_new_id = {}
    old_id_to_box = {}
    for object in tree.getroot().findall("object"):
        object_id = object.find("id").text
        points_elem = object.find("polygon").findall("pt")
        xmin, xmax = int(points_elem[0].find("x").text), int(points_elem[1].find("x").text)
        ymin, ymax = int(points_elem[0].find("y").text), int(points_elem[2].find("y").text)
        box = utils.structures.Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        if xmax-xmin < size_threshold or ymax-ymin < size_threshold:
            continue
        old_id_to_new_id[object_id] = len(old_id_to_new_id)
        old_id_to_box[object_id] = box
    for object in tree.getroot().findall("object"):
        object_id = object.find("id").text
        if object_id not in old_id_to_new_id:
            continue
        new_id = old_id_to_new_id[object_id]
        box = old_id_to_box[object_id]
        name = object.find("name").text
        parts_elem = object.find("parts")
        if parts_elem:
            hasparts_elem = parts_elem.find("hasparts")
            if hasparts_elem is not None and hasparts_elem.text is not None:
                hasparts = [old_id_to_new_id[old_id] for old_id in hasparts_elem.text.split(",") 
                        if int(old_id) in old_id_to_new_id]
            else:
                hasparts = []
            ispartof_elem = parts_elem.find("ispartof")
            if ispartof_elem is not None and ispartof_elem.text is not None:
                old_id = ispartof_elem.text
                ispartof = old_id_to_new_id[old_id] if old_id in old_id_to_new_id else -1
            else:
                ispartof = -1
        annotations.append(utils.structures.ObjectAnnotation(id=new_id, name=name, box=box, 
                image_file=image_file, ispartof=ispartof, hasparts=hasparts))
    return annotations

def read_image_info_in_annotation_file(
        filepath: str
    ) -> utils.structures.ImageInfo:
    tree = ET.parse(filepath)
    im_filename = tree.getroot().find("filename").text
    im_width = int(tree.getroot().find("imagesize").find("nrows").text)
    im_height = int(tree.getroot().find("imagesize").find("ncols").text)
    return utils.structures.ImageInfo(file=im_filename, width=im_width, height=im_height)