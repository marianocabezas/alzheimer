from operator import and_
import itertools
import numpy as np
from torch.utils.data.dataset import Dataset


class ImageCroppingDataset(Dataset):
    def __init__(self, source, target, lesion, mask, patch_size=32, overlap=32):
        # Init
        # Image and mask should be numpy arrays
        assert source.shape == mask.shape
        assert target.shape == mask.shape
        self.source = source
        self.target = target
        self.lesion = lesion
        self.mask = mask

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(self.mask.shape)
        patch_half = map(lambda p_length: p_length/2, patch_size)

        steps = map(lambda p_length: max(p_length - overlap, 1), patch_size)

        # Create bounding box and define
        min_bb = np.min(np.where(mask > 0), axis=-1)
        min_bb = map(
            lambda (min_i, p_len): min_i + p_len,
            zip(min_bb, patch_half)
        )
        max_bb = np.max(np.where(mask > 0), axis=-1)
        max_bb = map(
            lambda (max_i, p_len): max_i - p_len,
            zip(max_bb, patch_half)
        )

        dim_range = map(lambda t: np.arange(*t), zip(min_bb, max_bb, steps))
        self.patch_slices = map(
            lambda voxel: tuple(map(
                lambda (idx, p_len): slice(idx - p_len, idx + p_len),
                zip(voxel, patch_half)
            )),
            itertools.product(*dim_range)
        )

    def __getitem__(self, index):
        patch_slice = self.patch_slices[index]
        inputs = (
            np.expand_dims(self.source[patch_slice], 0),
            np.expand_dims(self.target[patch_slice], 0),
            np.expand_dims(self.lesion[patch_slice], 0),
            np.expand_dims(self.mask[patch_slice], 0)
        )
        target = np.expand_dims(self.target[patch_slice], 0)
        return inputs, target

    def __len__(self):
        return len(self.patch_slices)


class ImageListCroppingDataset(Dataset):
    def __init__(self, timepoints, lesion, mask, patch_size=32, overlap=32):
        # Init
        # Image and mask should be numpy arrays
        shape_comparisons = map(
            lambda (x, y): x.shape == y.shape,
            zip(timepoints[:-1], timepoints[1:])
        )
        assert reduce(and_, shape_comparisons)
        self.timepoints = timepoints
        timepoints_idx = range(len(timepoints))
        timepoints_combo = map(
            lambda i: map(
                lambda j: (i, j),
                timepoints_idx[i + 1:]
            ),
            timepoints_idx[:-1]
        )
        self.combos = np.concatenate(timepoints_combo, axis=0)
        self.lesion = lesion
        self.mask = mask

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(self.mask.shape)
        patch_half = map(lambda p_length: p_length/2, patch_size)

        steps = map(lambda p_length: max(p_length - overlap, 1), patch_size)

        # Create bounding box and define
        min_bb = np.min(np.where(mask > 0), axis=-1)
        min_bb = map(
            lambda (min_i, p_len): min_i + p_len,
            zip(min_bb, patch_half)
        )
        max_bb = np.max(np.where(mask > 0), axis=-1)
        max_bb = map(
            lambda (max_i, p_len): max_i - p_len,
            zip(max_bb, patch_half)
        )

        dim_range = map(lambda t: np.arange(*t), zip(min_bb, max_bb, steps))
        self.patch_slices = map(
            lambda voxel: tuple(map(
                lambda (idx, p_len): slice(idx - p_len, idx + p_len),
                zip(voxel, patch_half)
            )),
            itertools.product(*dim_range)
        )

    def __getitem__(self, index):
        n_slices = len(self.patch_slices)
        combo_idx = index / n_slices
        patch_idx = index % n_slices
        source = self.timepoints[self.combos[combo_idx, 0]]
        target = self.timepoints[self.combos[combo_idx, 1]]
        patch_slice = self.patch_slices[patch_idx]
        inputs_p = (
            np.expand_dims(source[patch_slice], 0),
            np.expand_dims(target[patch_slice], 0),
            np.expand_dims(self.lesion[patch_slice], 0),
            np.expand_dims(self.mask[patch_slice], 0)
        )
        target_p = np.expand_dims(target[patch_slice], 0)
        return inputs_p, target_p

    def __len__(self):
        return len(self.patch_slices) * len(self.combos)
