import torch
import torch.utils.data as data
import os
import json
import copy
import random

from datasets.decoder import decode
from datasets.transform import (random_short_side_scale_jitter,
                                random_crop, horizontal_flip,
                                uniform_crop)
from datasets.video_container import get_video_container


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    labels_class_map = {}
    index = 0
    for class_label in data['labels']:
        if class_label == '':
            continue
        class_labels_map[class_label] = index
        labels_class_map[index] = class_label
        index += 1
    return class_labels_map, labels_class_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
    return video_names, annotations


class UcfHmdb(data.Dataset):
    """
        ucf101 and hmdb51 dataset for evaluating representation transfer.
    """

    def __init__(self,
                 dataset_name,
                 root_path,
                 annotation_path,
                 subset,
                 video_tmpl='{}.avi',
                 testing=False,
                 num_frames=16,
                 sample_stride=4,
                 n_samples_for_each_video=10,
                 n_samples_for_each_frame=3,
                 crop_size=112,
                 num_retries=10
                 ):
        assert subset in ['training', 'validation'], 'Split {} not supported for {}'.format(
            subset, dataset_name)
        self._dataset_name = dataset_name
        self._subset = subset

        self._testing = testing
        if not self._testing:
            self._num_clip = 1
        else:
            self._num_clip = n_samples_for_each_video * n_samples_for_each_frame
        self._num_spatial_crops = n_samples_for_each_frame
        self._num_temporal_crops = n_samples_for_each_video

        self._num_frames = num_frames
        self._sample_stride = sample_stride

        self._crop_size = crop_size
        self._num_retries = num_retries

        print('Making {} dataset...'.format(dataset_name))
        self.make_dataset(root_path, annotation_path,
                          subset, video_tmpl, self._num_clip)

    def make_dataset(self, root_path, annotation_path, subset, video_tmpl, clip_num):
        data = load_annotation_data(annotation_path)
        video_names, annotations = get_video_names_and_annotations(
            data, subset)
        self.class_to_idx, self.idx_to_class = get_class_labels(data)

        self._data = []
        for vid, annotation in zip(video_names, annotations):
            video_path = os.path.join(root_path, video_tmpl.format(vid))
            if not os.path.exists(video_path):
                print("{} not exists".format(video_path))
            try:
                _ = get_video_container(video_path)
            except Exception as e:
                continue

            sample = {
                'video_path': video_path,
                'video_id': vid.split('/')[1],
                'label': self.class_to_idx[annotation['label']]
            }

            for k in range(clip_num):
                sample_j = copy.deepcopy(sample)
                sample_j['crop_index'] = k
                sample_j['video_mata'] = {}
                self._data.append(sample_j)
        print('Make {} dataset ({} clips)'.format(
            self._dataset_name, len(self._data)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self._data[index]['video_path']

        if not self._testing:
            temporal_sample_index = -1
            spatial_sample_index = -1
            if self._crop_size == 112:
                min_size, max_size = 128, 170
            elif self._crop_size == 224:
                min_size, max_size = 256, 340
        elif self._testing:
            temporal_sample_index = (self._data[index]['crop_index']
                                     // self._num_spatial_crops)
            spatial_sample_index = (self._data[index]['crop_index']
                                    % self._num_spatial_crops)
            min_size, max_size = self._crop_size, self._crop_size

        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = get_video_container(
                    path, multi_thread_decode=True)
            except Exception as e:
                print("Failed to load video from {} with error {}".format(path, e))

            if video_container is None:
                index = random.randint(0, len(self._data) - 1)
                continue

            frames = decode(video_container,
                            self._sample_stride,
                            self._num_frames,
                            temporal_sample_index,
                            self._num_temporal_crops,
                            video_meta=self._data[index]['video_mata'],
                            target_fps=30
                            )

            if frames is None:
                index = random.randint(0, len(self._data) - 1)
                continue

            frames = frames.float()
            # Normalization
            frames = frames / 255.0
            frames = frames - torch.tensor([0.45, 0.45, 0.45])
            frames = frames / torch.tensor([0.225, 0.225, 0.225])

            frames = frames.permute(3, 0, 1, 2)

            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_size,
                max_scale=max_size,
                crop_size=self._crop_size
            )
            if self._testing:
                target = self._data[index]['video_id']
            else:
                target = self._data[index]['label']
            return frames, target, index

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            if self._subset == 'training':
                frames = random_short_side_scale_jitter(
                    frames, min_scale, max_scale
                )
                frames = random_crop(frames, crop_size)
                frames = horizontal_flip(0.5, frames)
            else:
                frames = random_short_side_scale_jitter(
                    frames, min_scale, max_scale
                )
                frames = random_crop(frames, crop_size)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames = random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = uniform_crop(frames, crop_size, spatial_idx)
        return frames

    def get_idx_to_label(self):
        return self.idx_to_class

    def __len__(self):
        return len(self._data)


if __name__ == '__main__':
    pass
