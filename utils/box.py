import math
import utils.structures as structures

def adjust_to_square(
        box: structures.Box,
        im_width: int,
        im_height: int
    ) -> structures.Box:
    size = max(box.w, box.h)
    size = min(size, im_width, im_height) # crop to fit in image
    
    center_x = math.floor(box.xmin + box.w/2)
    xmin, xmax = extend_interval(center_x, val_limit=im_width, extended_size=size)
    center_y = math.floor(box.ymin+ box.h/2)
    ymin, ymax = extend_interval(center_y, val_limit=im_height, extended_size=size)
    return structures.Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

def extend_interval(val_center: int, val_limit: int, extended_size: int) -> tuple[int,int]:
    if val_center < val_limit/2: # center is in left part
        val_min = max(val_center - math.floor(extended_size/2), 0)
        val_max = val_min + extended_size
    else: # center is in right part
        val_max = min(val_center + math.ceil(extended_size/2), val_limit)
        val_min = val_max - extended_size
    return val_min, val_max