import os
import json
import random

import torch
import torch.utils.data as data
from transformers import DistilBertTokenizer

from datasets.decoder import decode
from datasets.transform import (random_short_side_scale_jitter,
                                random_crop, horizontal_flip,
                                uniform_crop)
from datasets.video_container import get_video_container


class Instagram(data.Dataset):
    """
        Instagram dataset for video text pair discrimination.
    """

    def __init__(self,
                 root_path,
                 list_file,
                 subset,
                 video_tmpl='trimmed_resize_{}.mp4',
                 num_frames=16,
                 sample_stride=4,
                 crop_size=112,
                 num_retries=10,
                 max_sentence_length=64
                 ):
        assert subset in [
            'training', 'validation'], 'Split {} not supported for Instagram'.format(subset)
        self._subset = subset

        self._num_frames = num_frames
        self._sample_stride = sample_stride

        self._crop_size = crop_size
        self._num_retries = num_retries

        print('Init tokenizer...')
        self._tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
        self._max_sentence_length = max_sentence_length

        print('Making dataset...')
        self.make_dataset(root_path, list_file, video_tmpl)

    def make_dataset(self, video_prefix_path, list_file, video_tmpl):
        self._data = []

        with open(list_file, 'r') as load_f:
            database = json.load(load_f)

        for vid, info in database.items():
            if info['subset'] == self._subset:
                video_path = os.path.join(
                    video_prefix_path, video_tmpl.format(vid))

                if not os.path.exists(video_path):
                    print("{} not exists".format(video_path))
                    continue

                sample = {
                    'video_path': video_path,
                    'video_id': vid,
                    'text': info['caption'],
                    'video_mata': {}
                }

                self._data.append(sample)
        print('Make {} dataset ({} clips)'.format(list_file, len(self._data)))

    @torch.no_grad()
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, text_embedding, index).
        """
        temporal_sample_index = -1
        spatial_sample_index = -1
        if self._crop_size == 112:
            min_size, max_size = 128, 170
        elif self._crop_size == 224:
            min_size, max_size = 256, 340

        for _ in range(self._num_retries):
            path = self._data[index]['video_path']

            text = self._data[index]['text']
            text_ids = torch.tensor(self._tokenizer.encode(text))
            if text_ids.shape[0] > 512:
                index = random.randint(0, len(self._data) - 1)
                continue
            # padding to fixed length
            text_embedding = torch.zeros(self._max_sentence_length)
            if self._max_sentence_length > text_ids.shape[0]:
                text_embedding[0:text_ids.shape[0]] = text_ids
            else:
                text_embedding = text_ids[0:self._max_sentence_length]
            text_embedding = text_embedding.long()

            video_container = None
            try:
                video_container = get_video_container(
                    path, multi_thread_decode=True)
            except Exception as e:
                print("Failed to load video from {} with error {}".format(path, e))

            if video_container is None:
                index = random.randint(0, len(self._data) - 1)
                continue

            # print('decode video')
            frames = decode(video_container,
                            self._sample_stride,
                            self._num_frames,
                            temporal_sample_index,
                            1,
                            video_meta=self._data[index]['video_mata'],
                            target_fps=30
                            )

            if frames is None:
                index = random.randint(0, len(self._data) - 1)
                continue

            frames = frames.float()
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

            return frames, text_embedding, index

    @torch.no_grad()
    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Modify from pySlowFast('https://github.com/facebookresearch/SlowFast')

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
            frames = random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = random_crop(frames, crop_size)
            frames = horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames = random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = uniform_crop(frames, crop_size, spatial_idx)
        return frames

    def __len__(self):
        return len(self._data)


if __name__ == '__main__':
    pass
