# Week 4: Speed Estimation and MTMC tracking

## Folder structure 
The code and data is structured as follows:<br>
   . <br>                      
   ├── aic19-track1-mtmc-train/<br>
   ├── detections/<br>
   ├── estimate_speed/<br>
   ├── metric_training.py<br>
   ├── metric_sweep.py<br>
   ├── README.md<br>
   ├── metric_visualization.ipynb<br>
   ├── create_metric_dataset.ipynb<br>
   ├── multicamera_tracking.ipynb<br>
   ├── sort.py<br>
   ├── utils.py<br>

In this structure:

* `aic19-track1-mtmc-train`: Folder containing the AIC challenge data.
* `detections`: Yolov8 bbox detections for every camera.
* `estimate_speed`: Folder containing all information to compute the speed of cars in a video (Task 1).
* `metric_training.py`: Code to train the siamese network in metric learning.
* `metric_sweep.py`: Code to perform W&B sweep in order to tune the siamese network.
* `metric_visualization.ipynb`: Notebook used to visualize the embeddings produced by the siamese network.
* `create_metric_dataset.ipynb`: Notebook to create the metric learning dataset.
* `multicamera_tracking.ipynb`: Self-contained notebook to perform MTMC tracking.
* `sort.py`: Imported script to create SORT tracker.
* `utils.py`: Python script with utilities for creating the metric learning dataset. 

## Running the code
Since the code is in _.ipynb_ format, it is required to have _Jupyter Notebook_ or any other program/text editor that can run this kind of file. We recommend using Visual Studio Code with the _Jupyter_ extension.

To replicate the submitted results and ensure that everything works as expected, simply use the __Run all__ button (or run the code blocks from top to bottom).
