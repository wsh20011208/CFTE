import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


def sample_symmetric_frame_indices(num_frames, temporal_offset=3):
    """Sample an exact, ordered triplet [t-k, t, t+k].

    For this Refined AdaptiveCMR ablation, k is fixed by the configuration and temporal
    reversal is not permitted. Therefore the returned order is always:
        source_past   = t - k
        driving       = t
        source_future = t + k
    """
    temporal_offset = int(temporal_offset)
    if temporal_offset < 1:
        raise ValueError("temporal_offset must be at least 1.")

    minimum_frames = 2 * temporal_offset + 1
    if num_frames < minimum_frames:
        raise ValueError(
            f"Video has {num_frames} frames, but exact +/-{temporal_offset} "
            f"sampling requires at least {minimum_frames} frames."
        )

    center_idx = np.random.randint(temporal_offset, num_frames - temporal_offset)
    return np.array(
        [center_idx - temporal_offset, center_idx, center_idx + temporal_offset],
        dtype=np.int64,
    )


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None,
                 min_temporal_offset=None, max_temporal_offset=None,
                 fixed_temporal_offset=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling

        # Exact symmetric temporal sampling for the ablation:
        # source_past=t-3, driving=t, source_future=t+3.
        if fixed_temporal_offset is not None:
            self.fixed_temporal_offset = max(1, int(fixed_temporal_offset))
        elif min_temporal_offset is not None and max_temporal_offset is not None:
            if int(min_temporal_offset) != int(max_temporal_offset):
                raise ValueError(
                    "Refined AdaptiveCMR requires a single fixed temporal offset; "
                    "min_temporal_offset and max_temporal_offset must be equal."
                )
            self.fixed_temporal_offset = max(1, int(min_temporal_offset))
        else:
            self.fixed_temporal_offset = 3
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            # Temporal order must remain past -> current -> future. The YAML sets
            # time_flip=False; this defensive override prevents accidental reversal
            # if another configuration is used later.
            augmentation_params = dict(augmentation_params)
            if augmentation_params.get('flip_param') is not None:
                augmentation_params['flip_param'] = dict(augmentation_params['flip_param'])
                augmentation_params['flip_param']['time_flip'] = False
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def _resolve_training_path(self, idx):
        """Resolve a training sample for both identity-style and direct filenames.

        This keeps ``id_sampling=True`` compatible with either:
          - identity-grouped names such as ``id#clip.mp4``; or
          - ordinary filenames such as ``clip.mp4`` without ``#``.

        Supported samples may be videos, concatenated frame images, or folders
        containing frame images.
        """
        entry = self.videos[idx]
        direct_path = os.path.join(self.root_dir, entry)
        supported_extensions = (
            '.mp4', '.mov', '.gif', '.png', '.jpg', '.jpeg'
        )

        # Some datasets list ordinary filenames rather than identity prefixes.
        # In that case, use the listed file/folder directly.
        if os.path.isdir(direct_path):
            return direct_path
        if (
            os.path.isfile(direct_path)
            and direct_path.lower().endswith(supported_extensions)
        ):
            return direct_path

        if self.id_sampling:
            identity = entry.split('#')[0]
            candidates = []
            for candidate in glob.glob(
                os.path.join(self.root_dir, identity + '*')
            ):
                if os.path.isdir(candidate):
                    candidates.append(candidate)
                elif (
                    os.path.isfile(candidate)
                    and candidate.lower().endswith(supported_extensions)
                ):
                    candidates.append(candidate)

            if not candidates:
                return None

            return str(np.random.choice(candidates))

        return direct_path if os.path.exists(direct_path) else None
    def _load_training_triplet(self, idx):
        """Load an exact [t-3, t, t+3] triplet.

        The center frame is sampled only from the valid range, so a center
        near either boundary is never selected. If the selected video itself
        has fewer than seven frames, another training video is sampled instead
        of aborting the DataLoader worker.
        """
        minimum_frames = 2 * self.fixed_temporal_offset + 1
        num_videos = len(self.videos)
        if num_videos == 0:
            raise RuntimeError('The training split contains no videos.')

        # First try the index requested by DataLoader. Subsequent attempts use
        # another random video. The bound avoids an infinite loop if the whole
        # split is invalid.
        max_attempts = max(20, 2 * num_videos)
        candidate_idx = int(idx) % num_videos

        for attempt in range(max_attempts):
            if attempt > 0:
                candidate_idx = int(np.random.randint(0, num_videos))

            path = self._resolve_training_path(candidate_idx)
            if path is None:
                continue

            video_name = os.path.basename(path)

            if os.path.isdir(path):
                frames = sorted(os.listdir(path))
                num_frames = len(frames)
                if num_frames < minimum_frames:
                    continue

                frame_idx = sample_symmetric_frame_indices(
                    num_frames, self.fixed_temporal_offset)
                video_array = [
                    img_as_float32(io.imread(os.path.join(path, frames[frame_id])))
                    for frame_id in frame_idx
                ]
            else:
                video_array = read_video(path, frame_shape=self.frame_shape)
                num_frames = len(video_array)
                if num_frames < minimum_frames:
                    continue

                frame_idx = sample_symmetric_frame_indices(
                    num_frames, self.fixed_temporal_offset)
                video_array = video_array[frame_idx]

            return video_array, video_name

        raise RuntimeError(
            f'Unable to find a training video with at least {minimum_frames} '
            f'frames after {max_attempts} attempts. Check the training split.'
        )

    def __getitem__(self, idx):
        if self.is_train:
            video_array, video_name = self._load_training_triplet(idx)
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
            video_name = os.path.basename(path)
            video_array = read_video(path, frame_shape=self.frame_shape)
            frame_idx = range(len(video_array))
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source_past = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            source_future = np.array(video_array[2], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            # Keep the historical key `source` as the past reference so that
            # old visualization / testing code remains compatible.
            out['source'] = source_past.transpose((2, 0, 1))
            out['source_past'] = source_past.transpose((2, 0, 1))
            out['source_future'] = source_future.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}