from operator import and_
import itertools
import numpy as np
from torch.utils.data.dataset import Dataset


def get_slices_bb(masks, patch_size, overlap):
        patch_half = map(lambda p_length: p_length/2, patch_size)
        steps = map(lambda p_length: max(p_length - overlap, 1), patch_size)

        if type(masks) is list:
            min_bb = map(lambda mask: np.min(np.where(mask > 0), axis=-1), masks)
            min_bb = map(
                lambda min_bb_i: map(
                    lambda (min_i, p_len): min_i + p_len,
                    zip(min_bb_i, patch_half)
                ),
                min_bb
            )
            max_bb = map(lambda mask: np.max(np.where(mask > 0), axis=-1), masks)
            max_bb = map(
                lambda max_bb_i: map(
                    lambda (max_i, p_len): max_i - p_len,
                    zip(max_bb_i, patch_half)
                ),
                max_bb
            )

            dim_ranges = map(
                lambda (min_bb_i, max_bb_i): map(
                    lambda t: np.arange(*t), zip(min_bb_i, max_bb_i, steps)
                ),
                zip(min_bb, max_bb)
            )

            patch_slices = map(
                lambda dim_range: map(
                    lambda voxel: tuple(map(
                        lambda (idx, p_len): slice(idx - p_len, idx + p_len),
                        zip(voxel, patch_half)
                    )),
                    itertools.product(*dim_range)
                ),
                dim_ranges
            )
        else:
            # Create bounding box and define
            min_bb = np.min(np.where(masks > 0), axis=-1)
            min_bb = map(
                lambda (min_i, p_len): min_i + p_len,
                zip(min_bb, patch_half)
            )
            max_bb = np.max(np.where(masks > 0), axis=-1)
            max_bb = map(
                lambda (max_i, p_len): max_i - p_len,
                zip(max_bb, patch_half)
            )

            dim_range = map(lambda t: np.arange(*t), zip(min_bb, max_bb, steps))
            patch_slices = map(
                lambda voxel: tuple(map(
                    lambda (idx, p_len): slice(idx - p_len, idx + p_len),
                    zip(voxel, patch_half)
                )),
                itertools.product(*dim_range)
            )

        return patch_slices


class ImagePairListCroppingDataset(Dataset):
    def __init__(self, source, target, lesions, patch_size=32, overlap=32):
        # Init
        # Image and mask should be numpy arrays
        # Init
        # Image and mask should be numpy arrays
        shape_comparisons = map(
            lambda (x, y, l): x.shape == y.shape and x.shape == l.shape,
            zip(source, target, lesions)
        )

        assert reduce(and_, shape_comparisons)

        self.source = source
        self.target = target
        self.lesions = lesions

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(self.lesions[0].shape)

        self.patch_slices = get_slices_bb(lesions, patch_size, overlap)
        self.max_slice = np.cumsum(map(len, self.patch_slices))

    def __getitem__(self, index):
        # We select the case
        case = np.min(np.where(self.max_slice > index))
        case_source = self.source[case]
        case_target = self.target[case]
        case_slices = self.patch_slices[case]
        case_lesion = self.lesions[case]

        # Now we just need to look for the desired slice
        slices = [0] + self.max_slice
        patch_idx = index - slices[case]
        patch_slice = case_slices[patch_idx]
        inputs_p = (
            np.expand_dims(case_source[patch_slice], 0),
            np.expand_dims(case_target[patch_slice], 0),
        )
        target_p = np.expand_dims(case_lesion[patch_slice], 0)

        return inputs_p, target_p

    def __len__(self):
        return self.max_slice[-1]


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

        self.patch_slices = get_slices_bb(mask, patch_size, overlap)

    def __getitem__(self, index):
        patch_slice = self.patch_slices[index]
        source = np.expand_dims(self.source[patch_slice], 0)
        target = np.expand_dims(self.source[patch_slice], 0)
        lesion = np.expand_dims(self.lesion[patch_slice], 0)
        mask = np.expand_dims(self.mask[patch_slice], 0)
        inputs = (
            source,
            target,
            lesion,
            mask
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

        self.patch_slices = get_slices_bb(mask, patch_size, overlap)

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


class ImagesListCroppingDataset(Dataset):
    def __init__(self, cases, lesions, masks, patch_size=32, overlap=32):
        # Init
        # Image and mask should be numpy arrays
        shape_comparisons = map(
            lambda case: map(
                lambda (x, y): x.shape == y.shape,
                zip(case[:-1], case[1:])
            ),
            cases
        )
        case_comparisons = map(
            lambda shapes: reduce(and_, shapes),
            shape_comparisons
        )
        assert reduce(and_, case_comparisons)

        self.n_cases = len(cases)
        self.cases = cases
        case_idx = map(lambda case: range(len(case)), cases)
        timepoints_combo = map(
            lambda timepoint_idx: map(
                lambda i: map(
                    lambda j: (i, j),
                    timepoint_idx[i + 1:]
                ),
                timepoint_idx[:-1]
            ),
            case_idx
        )
        self.combos = map(
            lambda combo: np.concatenate(combo, axis=0),
            timepoints_combo
        )
        self.lesions = lesions
        self.masks = masks

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(self.masks[0].shape)

        self.patch_slices = get_slices_bb(masks, patch_size, overlap)

        case_slices = map(
            lambda (p, c): len(p) * len(c),
            zip(self.patch_slices, self.combos)
        )

        self.max_slice = np.cumsum(case_slices)

    def __getitem__(self, index):
        # We select the case
        case = np.min(np.where(self.max_slice > index))
        case_timepoints = self.timepoints[case]
        case_slices = self.patch_slices[case]
        case_combos = self.combos[case]
        case_lesion = self.lesions[case]
        case_mask = self.masks[case]

        # Now we just need to look for the desired slice
        slices = [0] + self.max_slice
        index_corr = index - slices[case]
        n_slices = len(case_slices)
        combo_idx = index_corr / n_slices
        patch_idx = index_corr % n_slices
        source = case_timepoints[case_combos[combo_idx, 0]]
        target = case_timepoints[case_combos[combo_idx, 1]]
        patch_slice = case_slices[patch_idx]
        inputs_p = (
            source[patch_slice],
            target[patch_slice],
            case_lesion[patch_slice],
            case_mask[patch_slice]
        )
        target_p = target[patch_slice]
        return inputs_p, target_p

    def __len__(self):
        return self.max_slice[-1]


class ImageListDataset(Dataset):
    def __init__(self, cases, lesions, masks):
        # Init
        # Image and mask should be numpy arrays
        shape_comparisons = map(
            lambda case: map(
                lambda (x, y): x.shape == y.shape,
                zip(case[:-1], case[1:])
            ),
            cases
        )
        case_comparisons = map(
            lambda shapes: reduce(and_, shapes),
            shape_comparisons
        )
        assert reduce(and_, case_comparisons)

        self.cases = cases

        case_idx = map(lambda case: range(len(case)), cases)
        timepoints_combo = map(
            lambda timepoint_idx: map(
                lambda i: map(
                    lambda j: (i, j),
                    timepoint_idx[i + 1:]
                ),
                timepoint_idx[:-1]
            ),
            case_idx
        )
        self.combos = map(
            lambda combo: np.concatenate(combo, axis=0),
            timepoints_combo
        )

        self.lesions = lesions
        self.masks = masks

        min_bb = np.min(
            map(
                lambda mask: np.min(np.where(mask > 0), axis=-1),
                masks
            ),
            axis=0
        )
        max_bb = np.max(
            map(
                lambda mask: np.max(np.where(mask > 0), axis=-1),
                masks
            ),
            axis=0
        )
        self.bb = tuple(
            map(
                lambda (min_i, max_i): slice(min_i, max_i),
                zip(min_bb, max_bb)
            )
        )

        self.max_combo = np.cumsum([0] + map(len, self.combos))

    def __getitem__(self, index):
        # We select the case
        case = np.max(np.where(self.max_combo <= index))
        case_timepoints = self.cases[case]
        case_combos = self.combos[case]
        case_lesion = self.lesions[case]
        case_mask = self.masks[case]

        # Now we just need to look for the desired slice
        combo_idx = index - self.max_combo[case]

        source = case_timepoints[case_combos[combo_idx, 0]]
        target = case_timepoints[case_combos[combo_idx, 1]]
        source_bb = np.expand_dims(source[self.bb], axis=0)
        target_bb = np.expand_dims(target[self.bb], axis=0)
        lesion_bb = np.expand_dims(case_lesion[self.bb], axis=0)
        mask_bb = np.expand_dims(case_mask[self.bb], axis=0)
        inputs_bb = (
            source_bb,
            target_bb,
            lesion_bb,
            mask_bb
        )
        return inputs_bb, target_bb

    def __len__(self):
        return self.max_combo[-1]
