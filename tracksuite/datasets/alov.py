import os
import numpy as np
from torch.utils.data import Dataset


class ALOVDataSet(Dataset):

    def __init__(self, videos_root, annotations_root):
        self.videos_root = os.path.expanduser(videos_root)
        self.annotations_root = os.path.expanduser(annotations_root)
        self.frames = []
        self.annotations = []
        self._parse_dataset()

    def _parse_dataset(self):
        annotations_folder_list = os.listdir(self.annotations_root)
        annotations_folder_list.sort()
        i = 0
        for annonotations_folder in annotations_folder_list:
            videos_sub_folder = os.listdir(os.path.join(self.videos_root, annonotations_folder))
            videos_sub_folder.sort()
            for video in videos_sub_folder:
                video_path = os.path.join(self.videos_root, annonotations_folder, video)
                annotations_ann = os.path.join(self.annotations_root, annonotations_folder, video + '.ann')
                ann_list = self._load_annotations(annotations_ann)
                frame_idx = ann_list[:, 0] - 1
                frame_idx = frame_idx.astype(int)
                frames = self._get_frames_with_anns(video_path, frame_idx)
                # remove frame index
                ann_list = np.delete(ann_list, 0, 1)

                self._pair_frames_and_anns(frames,ann_list)

    def _pair_frames_and_anns(self, frames, ann_list):
        for i in range(len(frames) - 1):
            self.frames.append((frames[i], frames[i + 1]))
            self.annotations.append((ann_list[i], ann_list[i + 1]))

    @staticmethod
    def _get_frames_with_anns(video_path, frame_idx):
        frames = os.listdir(video_path)
        frames.sort()
        frames = np.array(frames)

        # discard frames without annotations
        frames = frames[frame_idx]

        # add video path to frames
        frames = np.core.defchararray.add(video_path + '/', frames)
        return frames

    @staticmethod
    def _load_annotations(ann):
        return np.loadtxt(ann)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        # TODO: add doc string

        """
        Returns a tuple. 
        
        Index 0 - a pair of images where the first image is the
        examplar image cropped at the bounding box and scaled to [227, 227] and the
        second image is the search image with the same cropping and scaling.

        Index 1 - the bounding box of the first image at index 0
        """
        return [[], []]
