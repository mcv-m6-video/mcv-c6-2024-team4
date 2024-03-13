# Week 2: Object Detection and Tracking

## Folder structure 
The code and data is structured as follows:
   
   .                 
   ├── data
   ├── TrackEval
   ├── results
   ├── data_utils.py
   ├── eval_utils.py
   ├── sort.py
   ├── run_experiments.py
   ├── detection_inference.ipynb
   ├── kalman_tracking.ipynb
   ├── tracking_yolo.ipynb
   ├── README.md
   ├── S04_C016_Annotations_CVAT.zip
   ├── S04_C016_Annotations_PASCALVOC.zip
   ├── dataset.yaml

In this structure:

* `data`: Folder containing S03_C010 video data for the project and its gt annotations.
* `TrackEval`: Folder dedicated to the evaluation of tracking algorithms.
* `results`: Folder containing the results from all three training strategies.
* `data_utils.py`: Python script with utilities for data handling and processing.
* `eval_utils.py`: Python script containing utilities for object detecion evaluation metrics.
* `sort.py`: Python script implementing the SORT (Simple Online and Realtime Tracking) algorithm.
* `run_experiments.py`: Python script for fine-tuning YOLO following the three different training strategies.
* `detection_inference.ipynb`: Notebook for performing inference YOLO object detection model.
* `kalman_tracking.ipynb`: Notebook dedicated to implementing Kalman filter for tracking.
* `tracking_yolo.ipynb`: Notebook integrating YOLO model for tracking.
* `README.md`: Markdown file providing an overview and instructions for the project.
* `S04_C016_Annotations_CVAT.zip`: ZIP file containing S04_C016 video annotations in CVAT format.
* `S04_C016_Annotations_PASCALVOC.zip`: ZIP file containing S04_C016 video annotations in PASCAL VOC format.
* `dataset.yaml`: Dataset configuration required for training YOLOv8 on our custom dataset.


## Running the code
Since the code is in _.ipynb_ format, it is required to have _Jupyter Notebook_ or any other program/text editor that can run this kind of file. We recommend using Visual Studio Code with the _Jupyter_ extension.

To replicate the submitted results and ensure that everything works as expected, simply use the __Run all__ button (or run the code blocks from top to bottom).
