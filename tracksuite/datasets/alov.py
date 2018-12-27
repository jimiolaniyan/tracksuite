import os
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from tracksuite.utils import crop_and_scale


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
                ann_list = self._convert_anns_to_bbox(ann_list)

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

    @staticmethod
    def _convert_anns_to_bbox(ann):
        idx_w = [0, 2, 4, 6]
        idx_h = [1, 3, 5, 7]
        left = np.min(ann[:, idx_w], axis=1)
        top = np.min(ann[:, idx_h], axis=1)
        right = np.max(ann[:, idx_w], axis=1)
        bottom = np.max(ann[:, idx_h], axis=1)

        return np.column_stack((left,top,right, bottom))

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

        prev_frame = cv.imread(self.frames[index][0])
        curr_frame = cv.imread(self.frames[index][1])

        prev_bb = self.annotations[index][0]
        curr_bb = self.annotations[index][1]

        exemplar, search, new_bb = crop_and_scale(prev_frame, curr_frame, prev_bb, curr_bb, (227, 227))

        return (exemplar, search), new_bb
