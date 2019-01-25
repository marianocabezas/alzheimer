import itertools
import numpy as np
from torch.utils.data.dataset import Dataset


class ImageCroppingDataset(Dataset):
    def __init__(self, source, target, lesion, mask, patch_size=32, overlap=10):
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

        steps = map(lambda p_length: max(p_length - overlap, 1), patch_size)

        # Create bounding box and define
        min_bb = np.min(np.where(mask > 0), axis=-1)
        min_bb = map(
            lambda (min_i, p_len): min_i + p_len / 2,
            zip(min_bb, patch_size)
        )
        max_bb = np.max(np.where(mask > 0), axis=-1)
        max_bb = map(
            lambda (max_i, p_len): max_i - p_len / 2,
            zip(max_bb, patch_size)
        )

        dim_range = map(lambda t: np.arange(*t), zip(min_bb, max_bb, steps))
        self.patch_slices = map(
            lambda voxel: tuple(map(
                lambda (idx, p_len): slice(idx - p_len / 2, idx + p_len / 2),
                zip(voxel, patch_size)
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
