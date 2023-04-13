import os
from dataclasses import dataclass, field
import csv

import numpy as np
import cv2

@dataclass
class Box:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def w(self) -> int:
        return self.xmax-self.xmin

    @property
    def h(self) -> int:
        return self.ymax-self.ymin

    def numpy(self) -> np.ndarray:
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])

@dataclass
class ObjectAnnotation:
    """Annotation for an object in an image.

    Attributes:
        id (int): ID of the object.
        name (str): name of the object (likely, its type).
        box (Box): Position in the image.
        image_file (str): file name of the image
        ispartof (int): ID of a container object in which this object belongs. If such container
                object does not exist, the value is -1.
        hasparts (list[int]): list of IDzs that this object contains. Can be empty.
    """
    id: int
    name: str
    box: Box
    image_file: str
    ispartof: int = -1
    hasparts: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.id = int(self.id)
        self.ispartof = int(self.ispartof)
        self.hasparts = list(map(int, self.hasparts))

@dataclass
class ImageInfo:
    file: str
    width: int
    height: int

    def read_image(self, directory: str) -> np.ndarray:
        """Reads the image in the given directory. Returns an image array with shape (W,H,C).
        """
        path_to_file = os.path.join(directory, self.file)
        if not os.path.exists(path_to_file):
            raise ValueError(f"{self.file} not in directory {directory}")
        return cv2.imread(path_to_file)

    @property
    def file_basename(self) -> str:
        return os.path.splitext(self.file)[0]

