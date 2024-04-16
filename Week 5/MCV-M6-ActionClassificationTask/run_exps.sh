#!/bin/bash


python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 1 --num_spatial_crops 1
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 2 --num_spatial_crops 1
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 3 --num_spatial_crops 1
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 5 --num_spatial_crops 1
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 10 --num_spatial_crops 1
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 1 --num_spatial_crops 2
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 2 --num_spatial_crops 2
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 3 --num_spatial_crops 2
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 5 --num_spatial_crops 2
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 10 --num_spatial_crops 2
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 1 --num_spatial_crops 3
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 2 --num_spatial_crops 3
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 3 --num_spatial_crops 3
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 5 --num_spatial_crops 3
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 10 --num_spatial_crops 3
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 1 --num_spatial_crops 5
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 2 --num_spatial_crops 5
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 3 --num_spatial_crops 5
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 5 --num_spatial_crops 5
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 10 --num_spatial_crops 5
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 1 --num_spatial_crops 10
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 2 --num_spatial_crops 10
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 3 --num_spatial_crops 10
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 5 --num_spatial_crops 10
sleep 5

python src/train_task31.py ../data/frames --annotations-dir ./data/hmdb51/testTrainMulti_601030_splits --num_segments 10 --num_spatial_crops 10
sleep 5


