""" Dataset class for HMDB51 dataset. """

import os
import random
from enum import Enum

from glob import glob, escape
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


class HMDB51Dataset(Dataset):
    """
    Dataset class for HMDB51 dataset.
    """

    class Split(Enum):
        """
        Enum class for dataset splits.
        """
        TEST_ON_SPLIT_1 = 1
        TEST_ON_SPLIT_2 = 2
        TEST_ON_SPLIT_3 = 3

    class Regime(Enum):
        """
        Enum class for dataset regimes.
        """
        TRAINING = 1
        TESTING = 2
        VALIDATION = 3

    CLASS_NAMES = [
        "brush_hair", "catch", "clap", "climb_stairs", "draw_sword", "drink", 
        "fall_floor", "flic_flac", "handstand", "hug", "kick", "kiss", "pick", 
        "pullup", "push", "ride_bike", "run", "shoot_ball", "shoot_gun", "situp", 
        "smoke", "stand", "sword", "talk", "turn", "wave", 
        "cartwheel", "chew", "climb", "dive", "dribble", "eat", "fencing", 
        "golf", "hit", "jump", "kick_ball", "laugh", "pour", "punch", "pushup", 
        "ride_horse", "shake_hands", "shoot_bow", "sit", "smile", "somersault", 
        "swing_baseball", "sword_exercise", "throw", "walk"
    ]


    def __init__(
        self, 
        videos_dir: str, 
        annotations_dir: str, 
        split: Split, 
        regime: Regime, 
        clip_length: int, 
        crop_size: int, 
        temporal_stride: int,
        num_segments: int, 
        num_spatial_crops: int
    ) -> None:
        """
        Initialize HMDB51 dataset.

        Args:
            videos_dir (str): Directory containing video files.
            annotations_dir (str): Directory containing annotation files.
            split (Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            regime (Regimes): Dataset regime (TRAINING, TESTING, VALIDATION).
            split (Splits): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            clip_length (int): Number of frames of the clips.
            crop_size (int): Size of spatial crops (squares).
            temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
        """
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.split = split
        self.regime = regime
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.temporal_stride = temporal_stride
        self.num_segments = num_segments
        self.num_spatial_crops = num_spatial_crops

        self.annotation = self._read_annotation()
        self.transform = self._create_transform()


    def _read_annotation(self) -> pd.DataFrame:
        """
        Read annotation files.

        Returns:
            pd.DataFrame: Dataframe containing video annotations.
        """
        split_suffix = "_test_split" + str(self.split.value) + ".txt"

        annotation = []
        for class_name in HMDB51Dataset.CLASS_NAMES:
            annotation_file = os.path.join(self.annotations_dir, class_name + split_suffix)
            df = pd.read_csv(annotation_file, sep=" ").dropna(axis=1, how='all') # drop empty columns
            df.columns = ['video_name', 'train_or_test']
            df = df[df.train_or_test == self.regime.value]
            df = df.rename(columns={'video_name': 'video_path'})
            df['video_path'] = os.path.join(self.videos_dir, class_name, '') + df['video_path'].replace('\.avi$', '', regex=True)
            df = df.rename(columns={'train_or_test': 'class_id'})
            df['class_id'] = HMDB51Dataset.CLASS_NAMES.index(class_name)
            annotation += [df]

        return pd.concat(annotation, ignore_index=True)


    def _create_transform(self) -> v2.Compose:
        """
        Create transform based on the dataset regime.

        Returns:
            v2.Compose: Transform for the dataset.
        """
        if self.regime == HMDB51Dataset.Regime.TRAINING:
            return v2.Compose([
                v2.RandomResizedCrop(self.crop_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return v2.Compose([
                v2.Resize((self.crop_size, self.crop_size)), # Shortest side of the frame to be resized to the given size
                #v2.CenterCrop(self.crop_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def get_num_classes(self) -> int:
        """
        Get the number of classes.

        Returns:
            int: Number of classes.
        """
        return len(HMDB51Dataset.CLASS_NAMES)


    def __len__(self) -> int:
        """
        Get the length (number of videos) of the dataset.

        Returns:
            int: Length (number of videos) of the dataset.
        """
        return len(self.annotation)


    def __getitem__(self, idx: int) -> tuple:
        """
        Get item (video) from the dataset.

        Args:
            idx (int): Index of the item (video).

        Returns:
            tuple: Tuple containing video, label, and video path.
        """
        df_idx = self.annotation.iloc[idx]

        # Get video path from the annotation dataframe and check if it exists
        video_path = df_idx['video_path']
        assert os.path.exists(video_path)

        # Read frames' paths from the video
        frame_paths = sorted(glob(os.path.join(escape(video_path), "*.jpg"))) # get sorted frame paths
        
        if self.regime == HMDB51Dataset.Regime.TRAINING: # training
            data = self.prepare_training_batch(frame_paths)
        else:
            data = self.prepare_inference_batch(frame_paths)

        # Get label from the annotation dataframe and make sure video was read
        label = df_idx['class_id']
        assert data is not None
        
        return data, label, video_path

    def prepare_training_batch(self, frame_paths):
        video_len = len(frame_paths)

        if video_len <= self.clip_length * self.temporal_stride:
            # Not enough frames to create the clip
            clip_begin, clip_end = 0, video_len
        else:
            # Randomly select a clip from the video with the desired length (start and end frames are inclusive)
            clip_begin = random.randint(0, max(video_len - self.clip_length * self.temporal_stride, 0))
            clip_end = clip_begin + self.clip_length * self.temporal_stride

        # Read frames from the video with the desired temporal subsampling
        video = None
        for i, path in enumerate(frame_paths[clip_begin:clip_end:self.temporal_stride]):
            frame = read_image(path)  # (C, H, W)
            if video is None:
                video = torch.zeros((self.clip_length, 3, frame.shape[1], frame.shape[2]), dtype=torch.uint8)
            video[i] = frame

        return video
    
    def distribute_crops(self, image_height, image_width, crop_size, n_crops):
        crops = []

        # If only one crop is requested, place it at the center
        if n_crops == 1:
            left = (image_width - crop_size) // 2
            top = (image_height - crop_size) // 2
            crops.append((left, top))
            return crops

        # Calculate the step size for distributing the crops
        step_x = max(1, (image_width - crop_size) // (n_crops // 2 + 1))
        step_y = max(1, (image_height - crop_size) // (n_crops // 2 + 1))

        # Initialize variables to keep track of remaining crops
        remaining_crops = n_crops

        # Iterate over the rows and columns to generate crop positions
        for i in range(n_crops // 2 + 1):
            for j in range(n_crops // 2 + 1):
                if remaining_crops <= 0:
                    break

                left = min(image_width - crop_size, max(0, j * step_x))
                top = min(image_height - crop_size, max(0, i * step_y))
                crops.append((left, top))
                
                remaining_crops -= 1

        return crops

    def prepare_inference_batch(self, frame_paths):
        video_len = len(frame_paths)
        segment_len = video_len // self.num_segments
        snippets = []
        frame = read_image(frame_paths[0])
        index_crops = self.distribute_crops(frame.shape[1], frame.shape[2], self.crop_size, self.num_spatial_crops)
        spatial_crops = [lambda x, size=size: v2.functional.crop(x, top=size[1], left=size[0], height=self.crop_size, width=self.crop_size) for size in index_crops]
        
        for seg_i in range(self.num_segments):
            start_seg = seg_i*segment_len
            end_seg = (seg_i+1)*segment_len

            if seg_i == self.num_segments - 1:
                # Adjust last segment's length to reach the end of the video
                segment_len = video_len - segment_len*(self.num_segments-1)
                end_seg = video_len

            if segment_len <= self.clip_length * self.temporal_stride:
                # Not enough frames to create the clip
                clip_begin, clip_end = 0, video_len
            else:
                # Randomly select a clip from the video with the desired length (start and end frames are inclusive)
                clip_begin = random.randint(start_seg, max(end_seg - self.clip_length * self.temporal_stride, start_seg))
                clip_end = clip_begin + self.clip_length * self.temporal_stride

            for cr_id, crop in enumerate(spatial_crops):
                # Read frames from the video with the desired temporal subsampling
                snippet = None
                for i, path in enumerate(frame_paths[clip_begin:clip_end:self.temporal_stride]):
                    if i == self.clip_length:
                        break
                    frame_1 = read_image(path)  # (C, H, W)
                    frame = crop(frame_1)
                    if snippet is None:
                        snippet = torch.zeros((self.clip_length, 3, frame.shape[1], frame.shape[2]), dtype=torch.uint8)
                    snippet[i] = frame

                snippets.append(snippet)

        return snippets
        


    def collate_fn(self, batch: list) -> dict:
        """
        Collate function for creating batches.

        Args:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched clips, labels, and paths.
        """
        # [(clip1, label1, path1), (clip2, label2, path2), ...] 
        #   -> ([clip1, clip2, ...], [label1, label2, ...], [path1, path2, ...])
        unbatched_clips, unbatched_labels, paths = zip(*batch)

        if self.regime == HMDB51Dataset.Regime.TRAINING:
            # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)
            transformed_clips = [self.transform(clip).permute(1, 0, 2, 3) for clip in unbatched_clips]
        else:
            # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)
            transformed_clips = [self.transform(snippet).permute(1, 0, 2, 3) for unbatched_snippets in unbatched_clips for snippet in unbatched_snippets]
        
        # Concatenate clips along the batch dimension: 
        # B * [(C, T, H, W)] -> B * [(1, C, T, H, W)] -> (B, C, T, H, W)
        batched_clips = torch.cat([d.unsqueeze(0) for d in transformed_clips], dim=0)

        return dict(
            clips=batched_clips, # (B, C, T, H, W)
            labels=torch.tensor(unbatched_labels), # (K,)
            paths=paths  # no need to make it a tensor
        )
