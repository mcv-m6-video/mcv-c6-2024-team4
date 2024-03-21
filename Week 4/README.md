# Week 3: Object Detection and Tracking

## Folder structure 
The code and data is structured as follows:<br>
   . <br>                      
   ├── aic19-track1-mtmc-train/<br>
   ├── detections/<br>
   ├── estimate_speed/<br>
   ├── metric_training.py<br>
   ├── README.md<br>
   ├── task1_1.ipynb<br>
   ├── metric_visualization.ipynb<br>
   ├── create_metric_dataset.ipynb<br>
   ├── multicamera_tracking.ipynb<br>
   ├── utils.py<br>

In this structure:

* `data`: Folder containing S03_C010 video data for the project and its gt annotations.
* `core`: Repository of RAFT.
* `pyflow`: Repository of pyflow.
* `deep_sort`: Repository of deepsort. It includes `run_experiments_deepsort.ipynb`, the notebook dedicated to implementing the tracking with DeepSORT algorithm.
* `utils.py`: Python script with utilities for creating the metric learning dataset.
* `evaluation.py`: Python script containing utilities for object detecion evaluation metrics.
* `task1_1.ipynb`: Notebook dedicated to implementing Block Matching algorithm.
* `task1_2.ipynb`: Notebook dedicated to implementing Off-the-shelf Optical Flow algorithms.
* `task1_3.ipynb`: Notebook dedicated to implementing tracking by adding Optical Flow.

## Running the code
Since the code is in _.ipynb_ format, it is required to have _Jupyter Notebook_ or any other program/text editor that can run this kind of file. We recommend using Visual Studio Code with the _Jupyter_ extension.

To replicate the submitted results and ensure that everything works as expected, simply use the __Run all__ button (or run the code blocks from top to bottom).
