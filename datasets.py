from operator import and_
import itertools
import numpy as np
from torch.utils.data.dataset import Dataset
from nibabel import load as load_nii
from data_manipulation.generate_features import get_mask_voxels


def get_image(name_or_image):
    if isinstance(name_or_image, basestring):
        return load_nii(name_or_image).get_data()
    elif isinstance(name_or_image, list):
        images = map(get_image, name_or_image)
        return np.stack(images, axis=0)
    else:
        return name_or_image


def get_slices_bb(masks, patch_size, overlap, filtered=False, min_size=0):
    patch_half = map(lambda p_length: p_length // 2, patch_size)
    steps = map(lambda p_length: max(p_length - overlap, 1), patch_size)

    if type(masks) is list:
        masks = map(get_image, masks)
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
                lambda t: np.concatenate([np.arange(*t), [t[1]]]),
                zip(min_bb_i, max_bb_i, steps)
            ),
            zip(min_bb, max_bb)
        )

        patch_slices = map(
            lambda dim_range: centers_to_slice(
                itertools.product(*dim_range), patch_half
            ),
            dim_ranges
        )

        if filtered:
            patch_slices = map(
                lambda (slices, mask): filter_size(slices, mask, min_size),
                zip(patch_slices, masks)
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
        patch_slices = centers_to_slice(
            itertools.product(*dim_range), patch_half
        )

        if filtered:
            patch_slices = filter_size(patch_slices, masks, min_size)

    return patch_slices


def centers_to_slice(voxels, patch_half):
    slices = map(
        lambda voxel: tuple(
            map(
                lambda (idx, p_len): slice(idx - p_len, idx + p_len),
                zip(voxel, patch_half)
            )
        ),
        voxels
    )
    return slices


def filter_bounds(voxels, min_bounds, max_bounds):
    filtered_voxels = filter(
        lambda v: reduce(
            and_, map(
                lambda (v_i, min_i, max_i): (v_i >= min_i) & (v_i <= max_i),
                zip(v, min_bounds, max_bounds)
            )
        ),
        voxels,
    )

    return filtered_voxels


def filter_size(slices, mask, min_size):
    filtered_slices = filter(
        lambda s_i: np.sum(mask[s_i] > 0) > min_size, slices
    )

    return filtered_slices


def get_balanced_slices(masks, patch_size, min_size=0):
    patch_half = map(lambda p_length: p_length // 2, patch_size)

    masks = map(get_image, masks)
    print('Getting lesion voxels')
    lesion_voxels = map(get_mask_voxels, masks)
    print('Getting other voxels')
    min_bb = map(lambda mask: np.min(np.where(mask > 0), axis=-1), masks)
    max_bb = map(lambda mask: np.max(np.where(mask > 0), axis=-1), masks)

    dim_ranges = map(
        lambda (min_bb_i, max_bb_i): map(
            lambda t: np.concatenate([np.arange(*t), [t[1]]]),
            zip(min_bb_i, max_bb_i, (1,) * 3)
        ),
        zip(min_bb, max_bb)
    )
    all_voxels = map(
        lambda dim_range: itertools.product(*dim_range), dim_ranges
    )

    min_shape = np.array(patch_half)
    max_shape = np.array(
        map(
            lambda (max_i, p_len): max_i - p_len,
            zip(masks[0].shape, patch_half)
        )
    )

    print('Filtering lesion bounds')
    flesion_voxels = map(
        lambda vox: filter_bounds(vox, min_shape, max_shape), lesion_voxels
    )

    print('Filtering other bounds')
    fall_voxells = map(
        lambda vox: filter_bounds(vox, min_shape, max_shape), all_voxels
    )
    fbck_voxels = map(
        lambda (vox, mask): filter(
            lambda vox_i: mask[vox_i] > 0,
            vox
        ),
        zip(fall_voxells, masks)
    )

    print('Slicing lesions')
    lesion_slices = map(
        lambda vox: centers_to_slice(vox, patch_half), flesion_voxels
    )
    print('Slicing other')
    bck_slices = map(
        lambda vox: centers_to_slice(vox, patch_half), fbck_voxels
    )

    print('Filtering other size')
    fbck_slices = map(
        lambda (slices, mask): filter_size(slices, mask, min_size),
        zip(bck_slices, masks)
    )

    print('Merging lesions and other')
    patch_slices = map(
        lambda (pos_s, neg_s): pos_s + map(
            lambda idx: neg_s[idx],
            np.random.permutation(len(neg_s))[:len(pos_s)]
        ),
        zip(lesion_slices, fbck_slices)
    )

    return patch_slices


def get_combos(cases, limits_only, step):
    if step is not None:
        if step < 1:
            step = 1

        case_idx = map(lambda case: [0, min(len(case) - 1, step)], cases)

    else:
        if limits_only:
            case_idx = map(lambda case: [0, len(case) - 1], cases)
        else:
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
    combos = map(
        lambda combo: np.concatenate(combo, axis=0),
        timepoints_combo
    )

    return combos


def get_mesh(shape):
    linvec = tuple(map(lambda s: np.linspace(0, s - 1, s), shape))
    mesh = np.stack(np.meshgrid(*linvec, indexing='ij')).astype(np.float32)
    return mesh


def assert_shapes(cases):
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


class GenericCroppingDataset(Dataset):
    def __init__(
            self,
            cases, labels=None, masks=None,
            patch_size=32, overlap=16, preload=False,
    ):
        # Init
        # Image and mask should be numpy arrays
        if preload:
            self.cases = map(get_image, cases)
        else:
            self.cases = cases
        self.labels = labels
        self.masks = masks

        data_shape = self.cases[0].shape

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        if self.masks is not None:
            self.patch_slices = get_slices_bb(
                self.masks, patch_size, overlap, filtered=True
            )
        elif self.labels is not None:
            self.patch_slices = get_slices_bb(
                self.labels, patch_size, overlap, filtered=True
            )
        else:
            data_single = map(lambda d: d[0] if len(d) > 1 else d, self.cases)
            self.patch_slices = get_slices_bb(data_single, patch_size, overlap)
        self.max_slice = np.cumsum(map(len, self.patch_slices))

    def __getitem__(self, index):
        # We select the case
        case_idx = np.min(np.where(self.max_slice > index))
        case = get_image(self.cases[case_idx])

        slices = [0] + self.max_slice.tolist()
        patch_idx = index - slices[case_idx]
        case_slices = self.patch_slices[case_idx]

        # We get the slice indexes
        slice_i = case_slices[patch_idx]

        inputs = np.expand_dims(case[slice_i], 0)

        if self.labels is not None:
            labels_patch = self.labels[case]
            labels = np.expand_dims(labels_patch, 0)

            return inputs, labels
        else:
            return inputs, case_idx, slice_i

    def __len__(self):
        return self.max_slice[-1]


class LongitudinalCroppingDataset(Dataset):
    def __init__(
            self,
            source, target, lesions, patch_size=32
    ):
        # Init
        # Image and mask should be numpy arrays
        shape_comparisons = map(
            lambda (x, y, l): x.shape == y.shape and x.shape[1:] == l.shape,
            zip(source, target, lesions)
        )

        assert reduce(and_, shape_comparisons)

        self.source = source
        self.target = target
        self.lesions = lesions
        data_shape = self.lesions[0].shape
        self.mesh = get_mesh(data_shape)

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(self.lesions[0].shape)

        self.patch_slices = get_slices_bb(
            lesions, patch_size, overlap=32, filtered=True, min_size=10
        )

        self.max_slice = np.cumsum(map(len, self.patch_slices))

    def __getitem__(self, index):
        # We select the case.
        case = np.min(np.where(self.max_slice > index))
        case_source = self.source[case]
        case_target = self.target[case]
        case_slices = self.patch_slices[case]
        case_lesion = self.lesions[case]

        # Now we just need to look for the desired slice
        slices = [0] + self.max_slice.tolist()
        patch_idx = index - slices[case]
        case_tuple = case_slices[patch_idx]

        # DF's initial mesh to generate a final deformation field.
        none_slice = (slice(None, None),)
        mesh = self.mesh[none_slice + case_tuple]
        source = case_source[none_slice + case_tuple]
        target = case_target[none_slice + case_tuple]

        inputs_p = (
            source,
            target,
            mesh,
            case_source

        )

        targets_p = (
            np.expand_dims(case_lesion[case_tuple], 0),
            target
        )

        return inputs_p, targets_p

    def __len__(self):
        return self.max_slice[-1]


class ImageListCroppingDataset(Dataset):
    def __init__(
            self,
            cases, lesions, masks,
            patch_size=32, overlap=16,
            limits_only=False,
            step=None,
    ):
        # Init
        # Image and mask should be numpy arrays
        assert_shapes(cases)

        self.cases = cases
        self.combos = get_combos(cases, limits_only, step)
        self.lesions = lesions
        self.masks = masks

        data_shape = self.lesions[0].shape

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)

        self.patch_slices = get_slices_bb(lesions, patch_size, overlap)
        self.mesh = get_mesh(data_shape)
        self.max_slice = np.cumsum(
            map(
                lambda (s, c): len(s) * len(c),
                zip(self.patch_slices, self.combos)
            )
        )

    def __getitem__(self, index):
        # We select the case
        case = np.min(np.where(self.max_slice > index))
        slices = [0] + self.max_slice.tolist()
        case_idx = index - slices[case]
        combo = self.combos[case]
        case_slices = self.patch_slices[case]

        n_slices = len(case_slices)
        combo_idx = case_idx // n_slices
        patch_idx = case_idx % n_slices

        case_source = self.cases[case][combo[combo_idx, 0]]
        case_target = self.cases[case][combo[combo_idx, 1]]
        slice_i = case_slices[patch_idx]

        case_lesion = self.lesions[case]
        case_mask = self.masks[case]

        mesh = self.mesh[(slice(None, None),) + slice_i]

        inputs_p = (
            np.expand_dims(case_source[slice_i], 0),
            np.expand_dims(case_target[slice_i], 0),
            np.expand_dims(case_lesion[slice_i], 0),
            np.expand_dims(case_mask[slice_i], 0),
            mesh,
            np.expand_dims(case_source, 0),
            np.expand_dims(case_lesion, 0),
        )
        targets_p = np.expand_dims(case_target[slice_i], 0)

        return inputs_p, targets_p

    def __len__(self):
        return self.max_slice[-1]


class ImageListDataset(Dataset):
    def __init__(self, cases, lesions, masks, limits_only=False, step=None):
        # Init
        # Image and mask should be numpy arrays
        assert_shapes(cases)

        self.cases = cases
        self.combos = get_combos(cases, limits_only, step)

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
