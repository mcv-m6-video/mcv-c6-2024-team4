# Week 4: Action classification on HMDB51

## Folder structure 
The code and data is structured as follows:<br>
   ```
    |____data/hmdb51/testTrainMulti_601030_splits
    |____videos**/
    |____frames**/
    |____src/
    |    |____datasets/
    |    |         |____HMDB51Dataset.py/
    |    |         |____HMDB51Dataset_task3.py/
    |    |         |____HMDB51Dataset_task4.py/
    |    |____models/
    |    |    |____ model_creator.py
    |    |____utils/
    |    |    |____ model_analysis.py
    |    |    |____ statistics.py
    |    |____train.py
    |    |____train_task2.py
    |    |____inference_task3.py
    |    |____train_task4.py
    |    |____sweep_task4.py
    |    |____visualize.ipynb
    |____README.md
    |____LICENSE
    |____requirements.txt
    |____run_exps.sh
   ```

In this structure:

* `data/hmdb51/testTrainMulti_601030_splits`: Modified Ground Truth annotations (detailed explanation in the `Custom groundtruth` section).
* `video`: Video data. **This folder can be created following the instructions in the `Download the dataset` section.
* `frames`: Video data, with the frames extracted. **This folder can be created following the instructions in the `Data preparation` section.
* `src/datasets/HMDB51Dataset.py`: Baseline dataset class.
* `src/datasets/HMDB51Dataset_task3.py`: Dataset class used in Task 3.
* `src/datasets/HMDB51Dataset_task4.py`: Dataset class used in Task 4.
* `src/models/model_creator.py`: Functions used for creating the model.
* `src/utils/model_analysis.py`: Functions to calculate the number of operations and parameters of the model.
* `src/utils/statistics.py`: Functions used for computing statistics.
* `src/train.py`: Script for training the baseline model. Detailed instructions in the `Run the baseline code` section.
* `src/train_task2.py`: Similar to train.py, but with metric logging to WandB.
* `src/inference_task3.py`: Script for runing the inference approaches described in Task 3.
* `src/train_task4.py`: Script for running the multi-view training proposed in Task 4.
* `src/sweep_task4.py`: Script for running a random hyperparameter gridsearch following Task 4's training approach.
* `src/visualize.ipynb`: Python Notebook used for visualizing the different plots regarding accuracy.
* `README.mb`: ReadME file.
* `LICENSE`: LICENSE file.
* `requirements.txt`: Requirements file with the dependencies required for running the code. Used in `Run the baseline code`. 
* `run_exps.sh`: Script for training the baseline model. Detailed instructions in the `Run the baseline code` section.

## Running the code

Make sure to follow the instructions in sections `Download the dataset`, `Data preparation` and `Run the baseline code` and have the folder structure shown above.

### Task 2 scripts
`src/train_task2.py`: Identical execution as the baseline code.

    ```
    $ python3 src/train_task2.py --help
        usage: train.py [-h] [--annotations-dir ANNOTATIONS_DIR]
                        [--clip-length CLIP_LENGTH] [--crop-size CROP_SIZE]
                        [--temporal-subsampling TEMPORAL_SUBSAMPLING]
                        [--model-name MODEL_NAME] [--load-pretrain]
                        [--optimizer-name OPTIMIZER_NAME] [--lr LR] [--epochs EPOCHS]
                        [--batch-size BATCH_SIZE] [--batch-size-eval BATCH_SIZE_EVAL]
                        [--validate-every VALIDATE_EVERY] [--num-workers NUM_WORKERS]
                        [--device DEVICE]
                        frames-dir

        Train a video classification model on HMDB51 dataset.

        positional arguments:
        frames_dir            Directory containing video files

        options:
        -h, --help            show this help message and exit
        --annotations-dir ANNOTATIONS_DIR
                                Directory containing annotation files
        --clip-length CLIP_LENGTH
                                Number of frames of the clips
        --crop-size CROP_SIZE
                                Size of spatial crops (squares)
        --temporal-subsampling TEMPORAL_SUBSAMPLING
                                Receptive field of the model will be (clip_length *
                                temporal_subsampling) / FPS
        --model-name MODEL_NAME
                                Model name as defined in models/model_creator.py
        --load-pretrain       Load pretrained weights for the model (if available)
        --optimizer-name OPTIMIZER_NAME
                                Optimizer name (supported: "adam" and "sgd" for now)
        --lr LR               Learning rate
        --epochs EPOCHS       Number of epochs
        --batch-size BATCH_SIZE
                                Batch size for the training data loader
        --batch-size-eval BATCH_SIZE_EVAL
                                Batch size for the evaluation data loader
        --validate-every VALIDATE_EVERY
                                Number of epochs after which to validate the model
        --num-workers NUM_WORKERS
                                Number of worker processes for data loading
        --device DEVICE       Device to use for training (cuda or cpu)
    ```

### Task 3 scripts
`src/inference_task3.py`: Similar execution as the baseline code, but with added parameters _num_segments_ and _num_spatial_crops_.

    ```
    $ python3 src/inference_task3.py --help
        usage: train.py [-h] [--annotations-dir ANNOTATIONS_DIR]
                        [--num_segments N_SEGMENTS] [--num_spatial_crops N_CROPS]
                        [--clip-length CLIP_LENGTH] [--crop-size CROP_SIZE]
                        [--temporal-subsampling TEMPORAL_SUBSAMPLING]
                        [--model-name MODEL_NAME] [--load-pretrain]
                        [--optimizer-name OPTIMIZER_NAME] [--lr LR] [--epochs EPOCHS]
                        [--batch-size BATCH_SIZE] [--batch-size-eval BATCH_SIZE_EVAL]
                        [--validate-every VALIDATE_EVERY] [--num-workers NUM_WORKERS]
                        [--device DEVICE]
                        frames-dir

        Train a video classification model on HMDB51 dataset.

        positional arguments:
        frames_dir            Directory containing video files

        options:
        -h, --help            show this help message and exit
        --annotations-dir ANNOTATIONS_DIR
                                Directory containing annotation files
        --num_segments N_SEGMENTS
                                Number of segments
        --num_spatial_crops N_CROPS
                                Number of spatial crops
        --clip-length CLIP_LENGTH
                                Number of frames of the clips
        --crop-size CROP_SIZE
                                Size of spatial crops (squares)
        --temporal-subsampling TEMPORAL_SUBSAMPLING
                                Receptive field of the model will be (clip_length *
                                temporal_subsampling) / FPS
        --model-name MODEL_NAME
                                Model name as defined in models/model_creator.py
        --load-pretrain       Load pretrained weights for the model (if available)
        --optimizer-name OPTIMIZER_NAME
                                Optimizer name (supported: "adam" and "sgd" for now)
        --lr LR               Learning rate
        --epochs EPOCHS       Number of epochs
        --batch-size BATCH_SIZE
                                Batch size for the training data loader
        --batch-size-eval BATCH_SIZE_EVAL
                                Batch size for the evaluation data loader
        --validate-every VALIDATE_EVERY
                                Number of epochs after which to validate the model
        --num-workers NUM_WORKERS
                                Number of worker processes for data loading
        --device DEVICE       Device to use for training (cuda or cpu)
    ```

### Task 4 scripts
`src/train_task4.py`: Similar execution as the baseline code, but with added parameters _num_segments_ and _agg_type_.

    ```
    $ python3 src/train_task4.py --help
        usage: train.py [-h] [--annotations-dir ANNOTATIONS_DIR]
                        [--num_segments N_SEGMENTS] [--agg_type AGG_FUN]
                        [--clip-length CLIP_LENGTH] [--crop-size CROP_SIZE]
                        [--temporal-subsampling TEMPORAL_SUBSAMPLING]
                        [--model-name MODEL_NAME] [--load-pretrain]
                        [--optimizer-name OPTIMIZER_NAME] [--lr LR] [--epochs EPOCHS]
                        [--batch-size BATCH_SIZE] [--batch-size-eval BATCH_SIZE_EVAL]
                        [--validate-every VALIDATE_EVERY] [--num-workers NUM_WORKERS]
                        [--device DEVICE]
                        frames-dir

        Train a video classification model on HMDB51 dataset.

        positional arguments:
        frames_dir            Directory containing video files

        options:
        -h, --help            show this help message and exit
        --annotations-dir ANNOTATIONS_DIR
                                Directory containing annotation files
        --num_segments N_SEGMENTS
                                Number of segments
        --agg_type AGG_FUN
                                Type of aggregation (mean/max)
        --clip-length CLIP_LENGTH
                                Number of frames of the clips
        --crop-size CROP_SIZE
                                Size of spatial crops (squares)
        --temporal-subsampling TEMPORAL_SUBSAMPLING
                                Receptive field of the model will be (clip_length *
                                temporal_subsampling) / FPS
        --model-name MODEL_NAME
                                Model name as defined in models/model_creator.py
        --load-pretrain       Load pretrained weights for the model (if available)
        --optimizer-name OPTIMIZER_NAME
                                Optimizer name (supported: "adam" and "sgd" for now)
        --lr LR               Learning rate
        --epochs EPOCHS       Number of epochs
        --batch-size BATCH_SIZE
                                Batch size for the training data loader
        --batch-size-eval BATCH_SIZE_EVAL
                                Batch size for the evaluation data loader
        --validate-every VALIDATE_EVERY
                                Number of epochs after which to validate the model
        --num-workers NUM_WORKERS
                                Number of worker processes for data loading
        --device DEVICE       Device to use for training (cuda or cpu)
    ```
    
`src/sweep_task4.py`: Same execution as `train_task4`, but some hyperparameters get overwritten by the gridsearch algorithm. Check the code to customize this setting.

    ```
    $ python3 src/sweep_task4.py --help
        usage: train.py [-h] [--annotations-dir ANNOTATIONS_DIR]
                        [--num_segments N_SEGMENTS] [--agg_type AGG_FUN]
                        [--clip-length CLIP_LENGTH] [--crop-size CROP_SIZE]
                        [--temporal-subsampling TEMPORAL_SUBSAMPLING]
                        [--model-name MODEL_NAME] [--load-pretrain]
                        [--optimizer-name OPTIMIZER_NAME] [--lr LR] [--epochs EPOCHS]
                        [--batch-size BATCH_SIZE] [--batch-size-eval BATCH_SIZE_EVAL]
                        [--validate-every VALIDATE_EVERY] [--num-workers NUM_WORKERS]
                        [--device DEVICE]
                        frames-dir

        Train a video classification model on HMDB51 dataset.

        positional arguments:
        frames_dir            Directory containing video files

        options:
        -h, --help            show this help message and exit
        --annotations-dir ANNOTATIONS_DIR
                                Directory containing annotation files
        --num_segments N_SEGMENTS
                                Number of segments
        --agg_type AGG_FUN
                                Type of aggregation (mean/max)
        --clip-length CLIP_LENGTH
                                Number of frames of the clips
        --crop-size CROP_SIZE
                                Size of spatial crops (squares)
        --temporal-subsampling TEMPORAL_SUBSAMPLING
                                Receptive field of the model will be (clip_length *
                                temporal_subsampling) / FPS
        --model-name MODEL_NAME
                                Model name as defined in models/model_creator.py
        --load-pretrain       Load pretrained weights for the model (if available)
        --optimizer-name OPTIMIZER_NAME
                                Optimizer name (supported: "adam" and "sgd" for now)
        --lr LR               Learning rate
        --epochs EPOCHS       Number of epochs
        --batch-size BATCH_SIZE
                                Batch size for the training data loader
        --batch-size-eval BATCH_SIZE_EVAL
                                Batch size for the evaluation data loader
        --validate-every VALIDATE_EVERY
                                Number of epochs after which to validate the model
        --num-workers NUM_WORKERS
                                Number of worker processes for data loading
        --device DEVICE       Device to use for training (cuda or cpu)
    ```
## Download the dataset

0. Before downloading any data, install the `unrar` and `ffmpeg` packages with your package manager. In Debian-based distros (e.g., Ubuntu), this is done with the following command: `sudo apt install unrar ffmpeg`.

1. Now, download the videos from the authors' website and decompress them with `unrar`:

    ```bash
        # Download the compressed dataset
        $ wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
        # Extract the file
        $ unrar x hmdb51_org.rar
        # Rename the decompressed folder from hmdb51_org -> videos
        $ mv hmdb51_org videos
        # Inspect the contents of the uncompressed folder
        $ ls videos
    ```

You should be seeing 51 folders, one for each action category (`brush_hair`, `cartwheel`, etc) containing video data (`.avi` files):

    ```
        videos/
        |____ brush_hair/
        |     |____ April_09_brush_hair_u_nm_np1_ba_goo_0.avi
        |     |____ April_09_brush_hair_u_nm_np1_ba_goo_1.avi
        |     |____ ...
        |
        |____ cartwheel/
        |     |____ ...
        |
        |____ ...
    ```

However, for our training procedure, it'll be much more convenient having the videos converted to frames first. Frames are faster to read than video and allow selective access depending on how we decide to sample clips.

## Data preparation

To extract the frames in a directory named `frames`, we can run this long one-line command:

    ```bash
    # Make sure you are in the same directory that contains the videos/ folder
    $ find videos -name "*.avi" -print0 | xargs -0 -I {} sh -c 'original="{}"; modified=$(echo "$original" | sed -e "s|videos|frames|" | sed -e "s|\.[^.]*$||"); mkdir -p $modified; ffmpeg -i $original -loglevel error $modified/frame%05d.jpg'
    ```
    
If run correctly, such command will create  `frames/` directory with the same structure as `videos/`, but replacing each video file by a directory containing the frames (in .jpg format). It might take from 30' to an hour to extract the frames (depending on your CPU).

Graphically:

    ```
    frames/
    |____ brush_hair/
    |     |____ April_09_brush_hair_u_nm_np1_ba_goo_0/
    |     |     |____ frame00001.jpg
    |     |     |____ frame00002.jpg
    |     |     |____ frame00003.jpg
    |     |     |____ ...
    |     |____ April_09_brush_hair_u_nm_np1_ba_goo_1/
    |           |____ frame00001.jpg
    |           |____ frame00002.jpg
    |           |____ frame00003.jpg
    |           |____ ...
    |
    |____ cartwheel/
    |     |____ ...
    |
    |____ ...
    ```

## Custom groundtruth

Do not download and use the groundtruth annotations from the authors' webpage, as we will be using a modified version that you'll find in `data/hmbd51/testTrainMulti_601030_splits` directory of this same repository.

Differently from the original groundtruth, we will reserve part of the training videos for validation. In particular, instead of having 70% training and 30% testing data, we will have 60% training, 10% validation, and 30% testing.

Then, just take into account that HMDB51 was thought to be evaluated in 3-fold cross validation. So you will see 3 different splits, namely split1, split2, and split3. In the provided annotations, this splits are done in a separate file for each action label (i.e, `<action_label>_test_split<1, 2, or 3>.txt`). However, we will focus on split1 only (ignore split2 and split3 files). Go and examine any `<action_label>_test_split1.txt` file and you'll find there's a line per video. Each line has the video name followed by an integer that represents the partition (train = 1, validation = 3 or test = 2) for this particular split.

## Run the baseline code

0. You'll also need to install the required Python dependencies. These are in the `requirements.txt` file. Assuming you are using PIP, you can then just run:

    ```bash
    $ pip3 install -r requirements.txt
    ```

1. Finally, the baseline can be run executing the `src/train.py` script, which expects one positional argument (the directory containing the frames that we've created before), but accepts other multiple arguments:

    ```
    $ python3 src/train.py --help
        usage: train.py [-h] [--annotations-dir ANNOTATIONS_DIR]
                        [--clip-length CLIP_LENGTH] [--crop-size CROP_SIZE]
                        [--temporal-subsampling TEMPORAL_SUBSAMPLING]
                        [--model-name MODEL_NAME] [--load-pretrain]
                        [--optimizer-name OPTIMIZER_NAME] [--lr LR] [--epochs EPOCHS]
                        [--batch-size BATCH_SIZE] [--batch-size-eval BATCH_SIZE_EVAL]
                        [--validate-every VALIDATE_EVERY] [--num-workers NUM_WORKERS]
                        [--device DEVICE]
                        frames-dir

        Train a video classification model on HMDB51 dataset.

        positional arguments:
        frames_dir            Directory containing video files

        options:
        -h, --help            show this help message and exit
        --annotations-dir ANNOTATIONS_DIR
                                Directory containing annotation files
        --clip-length CLIP_LENGTH
                                Number of frames of the clips
        --crop-size CROP_SIZE
                                Size of spatial crops (squares)
        --temporal-subsampling TEMPORAL_SUBSAMPLING
                                Receptive field of the model will be (clip_length *
                                temporal_subsampling) / FPS
        --model-name MODEL_NAME
                                Model name as defined in models/model_creator.py
        --load-pretrain       Load pretrained weights for the model (if available)
        --optimizer-name OPTIMIZER_NAME
                                Optimizer name (supported: "adam" and "sgd" for now)
        --lr LR               Learning rate
        --epochs EPOCHS       Number of epochs
        --batch-size BATCH_SIZE
                                Batch size for the training data loader
        --batch-size-eval BATCH_SIZE_EVAL
                                Batch size for the evaluation data loader
        --validate-every VALIDATE_EVERY
                                Number of epochs after which to validate the model
        --num-workers NUM_WORKERS
                                Number of worker processes for data loading
        --device DEVICE       Device to use for training (cuda or cpu)
    ```

If not specified, default values should allow you to run the script without issues.

Check the implementation to understand the different parts of the code.
