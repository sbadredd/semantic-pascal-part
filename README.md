# Semantic Pascal-Part Feature Generation

The Semantic Pascal-Part Feature Generation code generates feature vectors for each bounding box in the [Semantic PASCAL-Part](https://github.com/ivanDonadello/semantic-PASCAL-Part) neuro-symbolic dataset. These feature vectors are the result of the Region of Interest embedding by the FasterRCNN architecture as described in [Ren et al., 2017](https://arxiv.org/abs/1506.01497).

## Task Description
Semantic Pascal-Part is introduced by [Donadello et al., 2017](https://arxiv.org/abs/1705.08968) for a classification task that involves training two neural models. The first model predicts the type of an object within a bounding box `x`, using a multi-class, single-label classifier. This model allows us to answer questions such as `isbottle(x)`, `iscap(x)`, etc. The second model is a binary relation predictor that can determine if one bounding box `x` is a part of another bounding box `y`, using the `ispartof(x,y)` predicate.

In addition to learning via groundtruth examples (as usual in Machine Learning), the learning process can also use a set of mereological constraints that relate to the types and their meanings. These constraints can be used to reason about relationships between objects and their parts. For example:
```css
forall x,y ((isbottle(x) AND ispartof(y,x)) -> (iscap(y) OR isbody(x)))
forall x,y ((iscap(x) AND ispartof(x,y)) -> isbottle(y))
```

By using these constraints, the model can learn to reason about relationships between objects and their parts, even when there are fewer ground truth examples available.

Overall, Semantic Pascal-Part is a valuable dataset for exploring the use of logical constraints in guiding the learning process of neural models.

## Motivation for the New Feature Set

Previous works featurized the bounding boxes using the object class predictions produced by an object detector trained on the PASCAL-Part dataset [[Donadello et al., 2017](https://arxiv.org/abs/1705.08968); [van Krieken et al., 2019](https://arxiv.org/abs/1908.04700)]. However, this meant that the NeSy model only corrected the predictions of the detector. To increase the difficulty of the task, we generate the feature vectors using latent vectors of 1024 features, which are produced by intermediate layers of a pre-trained FasterRCNN model (specifically, the backbone + ROIPooler + box head). This means that the NeSy model must learn the final layers of the object classifier as well.

For those interested in an object detector architecture trained end-to-end without any pre-training, please refer to [Manigrasso et al., 2021](https://arxiv.org/abs/2107.01877).

## Requirements

Please refer to `requirements.txt`. We have specified the versions of each package used in development for reproducibility purposes. Similar versions should run fine.

## Usage
Note: Please run the scripts using this repository as your python active directory.

### Step 1: Download the Data
1. Download the original data from [https://zenodo.org/record/5878773#.YegfKiwo-qB](https://zenodo.org/record/5878773#.YegfKiwo-qB). This dataset contains the refined images and annotations (e.g., small specific parts are merged into bigger parts) of the PASCAL-Part dataset in Pascal-voc style.
2. Unzip the files to the `data/` folder in the appropriate subfolders.

The resulting directory structure should look like this:
```bash
data/
├── Annotations_test/  # contains the test set annotations in .xml format.
├── Annotations_trainval/  # contains the train and validation set annotations in .xml format. 
├── JPEGImages_test/  # contains the test set images in .jpg format.
├── JPEGImages_trainval/  # contains the train and validation set images in .jpg format.
├── test.txt  # contains the 2416 image filenames in the test set.
├── trainval.txt  # contains the 7687 image filenames in the train and validation set.
└── classes.csv  # contains the class IDs (row index) associated with each class. This file is already included in this repository.
```

### Step 2: Preprocessing
(optional) For increased readability, indent the xml files by running the following command:
```python
python scripts/preprocess/indent_xml_files.py --data test
```

You can substitute `test` with `trainval` to process either of the data.

### Step 3: Generate the Feature Vectors
To generate feature vectors for each bounding box using FasterRCNN, run the following command:

```bash
python scripts/compute/get_faster_rcnn_roi_features.py --data test
```

The output will be saved in raw `.csv` files located in either `output/raw/trainval` or `output/raw/test`. These files contain the following information:
* `features.csv`: contains the bounding boxes and their associated features, identified by their row number. Each row specifies the image that contains the bounding box (column 0), the bounding box coordinates (columns 1024 to 1027: xmin, ymin, xmax, ymax), and the latent vector from the FasterRCNN architecture (columns 1 to 1023). Note that row counts start from 0.
* `partOf.csv`: contains partOf(x,y) relations, where x is the row number of the bounding box and y is the value written in the row. If y is -1, then x is not a part of any object. Row counts start from 0.
* `types.csv`: contains the class ID (value in the row) associated with each bounding box (row number). Row counts start from 0.

### Step 4: Convert CSV Files to HDF5 Format (Optional, but Recommended)
For an efficient pipeline, we recommend converting the raw .csv files to HDF5 format by running:
```bash
python scripts/postprocess/process_output_to_hdf5.py --data test
```
This will convert the raw .csv files into HDF5 format, which is a more efficient file format for large datasets.

## HDF5 Files

The hdf5 files can be found in `output/processed/trainval` or `output/processed/test`. 
The dataset consists of two files:
* The `box_features.hdf5` file, which is used to store information about boxes and their associated attributes, including position and region of interest (ROI) features.
* The `pairs_partof.hdf5` file, which is used to store pair-wise relationships between boxes, specifically whether one box is part of another.

### `box_features.hdf5`
The file is organized in a hierarchical structure, with each box having a unique box_id as its primary key.
```vbnet
box_features.hdf5
│
└── box_id (group)
    ├── attrs (attributes)
    │   ├── "type" (string)
    │   ├── "pic" (integer)
    │   └── "partof" (integer)
    ├── "position" (dataset)
    └── "roi_features" (dataset)
```

#### Description of Fields
* `box_id` (group): A unique identifier for each box. This is used as the primary key for accessing and storing information about each box in the file.
* `attrs` (attributes): A collection of metadata attributes associated with a `box_id`.
    * `"type"` (string): The type of box, for example, 'chair' or 'screen'.
    * `"pic"` (integer): A reference to the image that the box is part of, using a unique identifier for each image.
    * `"partof"` (integer): A reference to another box (if any) that the current box is a part of, using the unique identifier of the parent box. If the current box is not part of any other box, this value is set to -1.
* `"position"` (dataset): A 1D array of float values representing the coordinates (xmin, ymin, xmax, ymax) of the box. The length of this array is 4.
* `"roi_features"` (dataset): A 1D array of float values representing the ROI features of the box, generated using FasterRCNN. The length of this array is 1024.

#### Usage Example

Here's an example of how to access the box_features.hdf5 file using the h5py library in Python:
```python
from __future__ import annotations
import dataclasses
import tqdm
import h5py
import numpy as np

@dataclasses.dataclass
class Box:
    id_: int
    type_: str
    pic: int
    partof_id: int
    roi_features: np.ndarray
    position: np.ndarray

box_data: list[Box] = []

with h5py.File("box_features.hdf5", "r") as f:
    all_box_ids = list(f.keys())
    
    for box_id in tqdm.tqdm(all_box_ids, desc="Reading box ids"):
        box_data.append(
            Box(id_=box_id,
                type_=f[box_id].attrs["type"],
                pic=f[box_id].attrs["pic"],
                partof_id=f[box_id].attrs["partof"],
                roi_features=f[box_id]["roi_features"][()],
                position=f[box_id]["position"][()]
            )
        )
```

Also, note that you can access the list of all box ids using:
```python
box_ids = list(f.keys())
```

### `pairs_partof.hdf5`

The structure of the `pairs_partof.hdf5` file is as follows:
```vbnet
pairs_partof.hdf5
│
└── box1_id (group)
    └── box2_id (group)
        └── attrs (attributes)
            └── "ispartof" (boolean)
```

Notice that we only store pair data for `box1_id` and `box2_id` values that appear in the same image.

#### Description of Fields

* `box1_id` (group): A unique identifier for the first box in the pair. This is used as the primary key for accessing and storing pair-wise relationships.
* `box2_id` (group): A unique identifier for the second box in the pair.
* `attrs` (attributes): A collection of metadata attributes associated with a pair of boxes (`box1_id` and `box2_id`).
    * `"ispartof"` (boolean): Indicates whether box1 is part of box2. A value of True means box1 is part of box2, while a value of False means it is not.

#### Usage Example
Here's an example of how to access the `pairs_partof.hdf5` file using the h5py library in Python:

```python
from __future__ import annotations
import dataclasses
import h5py

class Pair:
    box1_id: int
    box2_id: int
    ispartof: bool
    
pair_data: list[Pair] = []

# Replace this with the actual box data (List of box objects with id_ attribute)
# See above Example
box_data = []

# Open the file
with h5py.File("pairs_partof.hdf5", "r") as f:
    for box1 in box_data:
        box1_group = f[str(box1.id_)]
        for box2_id in box1_group.keys():
            ispartof = box1_group[box2_id].attrs["ispartof"]
            paired_data.append(PairedData(box1.id_, int(box2_id), ispartof))
```

## Miscellaneous
In addition to the main code for generating feature vectors for each bounding box in the Semantic PASCAL-Part dataset, we also provide two useful scripts:
* `scripts/preprocess/crop_bbox_images.py`: This script can be used to crop all bounding boxes in the images.
* `scripts/postprocess/show_image_and_boxes.py`: This script can be used to plot an image and highlight the bounding boxes appearing in that image.
These scripts can be helpful for visualizing the bounding boxes in the dataset and for preparing cropped images for downstream tasks.