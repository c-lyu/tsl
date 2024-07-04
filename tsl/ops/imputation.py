from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from tsl import logger
from tsl.datasets.prototypes import TabularDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.utils.python_utils import ensure_list


def sample_mask(
    shape,
    p: float = 0.002,
    p_noise: float = 0.0,
    p_black: float = 0.0,
    max_seq: int = 1,
    min_seq: int = 1,
    max_seq_black: int = 1,
    min_seq_black: int = 1,
    rng: np.random.Generator = None,
    verbose: bool = True,
):
    assert 0 <= p <= 1, f"p must be between 0 and 1, got {p}"
    assert 0 <= p_noise <= 1, f"p_noise must be between 0 and 1, got {p_noise}"
    assert 0 <= p_black <= 1, f"p_black must be between 0 and 1, got {p_black}"
    assert (
        min_seq_black <= max_seq_black
    ), f"min_seq_black must be less than or equal to max_seq_black, got {min_seq_black} and {max_seq_black}"

    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    if verbose:
        logger.info(f"Generating mask with base p={p}")

    mask = rand(shape) < p

    # Generate blackout-style mask
    if p_black > 0:
        blackout_mask = np.zeros(shape[0], dtype=bool)
        idx = 0
        while idx < shape[0]:
            if rand() < p_black:
                blackout_len = randint(min_seq_black, max_seq_black + 1)
                end = min(idx + blackout_len, shape[0])
                blackout_mask[idx:end] = True
                idx = end
            else:
                idx += 1
        mask |= blackout_mask[:, None, None]

    # Generate fault mask for each sensor
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True

    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype("uint8")


def missing_val_lens(mask):
    m = np.concatenate(
        [
            np.zeros((1, mask.shape[1])),
            (~mask.astype("bool")).astype("int"),
            np.zeros((1, mask.shape[1])),
        ]
    )
    mdiff = np.diff(m, axis=0)
    lens = []
    for c in range(m.shape[1]):
        (mj,) = mdiff[:, c].nonzero()
        diff = np.diff(mj)[::2]
        lens.extend(list(diff))
    return lens


def to_missing_values_dataset(
    dataset: TabularDataset, eval_mask: np.ndarray, inplace: bool = True
):
    assert isinstance(dataset, TabularDataset)
    if not inplace:
        dataset = deepcopy(dataset)

    # Dynamically inherit from MissingValuesDataset
    bases = tuple([dataset.__class__, MissingValuesMixin])
    cls_name = "MissingValues%s" % dataset.__class__.__name__
    dataset.__class__ = type(cls_name, tuple(bases), {})
    # Change dataset name
    dataset.name = "MissingValues%s" % dataset.name

    dataset.set_eval_mask(eval_mask)

    return dataset


def add_missing_values(
    dataset: TabularDataset,
    p_noise=0.05,
    p_fault=0.01,
    p_black=0.0,
    min_seq=1,
    max_seq=10,
    min_seq_black=1,
    max_seq_black=10,
    seed=None,
    inplace=True,
    cache_dir=None,
):
    if seed is None:
        seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random = np.random.default_rng(seed)

    # Compute evaluation mask
    shape = (dataset.length, dataset.n_nodes, dataset.n_channels)
    eval_mask = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        eval_mask_path = cache_dir / f"eval_mask_{shape[0]},{shape[1]}-{p_fault}-{p_noise}-{p_black}-{min_seq}-{max_seq}-{min_seq_black}-{max_seq_black}-{seed}.npy"
        if eval_mask_path.exists():
            eval_mask = np.load(eval_mask_path)
    if eval_mask is None:
        eval_mask = sample_mask(
            shape,
            p=p_fault,
            p_noise=p_noise,
            p_black=p_black,
            min_seq=min_seq,
            max_seq=max_seq,
            min_seq_black=min_seq_black,
            max_seq_black=max_seq_black,
            rng=random,
        )
        if cache_dir is not None:
            np.save(eval_mask_path, eval_mask)

    # Convert to missing values dataset
    dataset = to_missing_values_dataset(dataset, eval_mask, inplace)

    # Store evaluation mask params in dataset
    dataset.p_fault = p_fault
    dataset.p_noise = p_noise
    dataset.p_black = p_black
    dataset.min_seq = min_seq
    dataset.max_seq = max_seq
    dataset.min_seq_black = min_seq_black
    dataset.max_seq_black = max_seq_black
    dataset.seed = seed
    dataset.random = random

    return dataset


def prediction_dataframe(y, index, columns=None, aggregate_by="mean"):
    """Aggregate batched predictions in a single DataFrame.

    Args:
        y (list or np.ndarray): The list of predictions.
        index (list or np.ndarray): The list of time indexes coupled with
            the predictions.
        columns (list or pd.Index): The columns of the returned DataFrame.
        aggregate_by (str or list): How to aggregate the predictions in case
            there are more than one for a step.

            - `mean`: take the mean of the predictions;
            - `central`: take the prediction at the central position, assuming
              that the predictions are ordered chronologically;
            - `smooth_central`: average the predictions weighted by a gaussian
              signal with std=1.

    Returns:
        pd.DataFrame: The evaluation mask for the DataFrame.
    """
    dfs = [
        pd.DataFrame(data=data.reshape(data.shape[:2]), index=idx, columns=columns)
        for data, idx in zip(y, index)
    ]
    df = pd.concat(dfs)
    preds_by_step = df.groupby(df.index)
    # aggregate according passed methods
    aggr_methods = ensure_list(aggregate_by)
    dfs = []
    for aggr_by in aggr_methods:
        if aggr_by == "mean":
            dfs.append(preds_by_step.mean())
        elif aggr_by == "central":
            dfs.append(preds_by_step.aggregate(lambda x: x[int(len(x) // 2)]))
        elif aggr_by == "smooth_central":
            from scipy.signal import gaussian

            dfs.append(
                preds_by_step.aggregate(
                    lambda x: np.average(x, weights=gaussian(len(x), 1))
                )
            )
        elif aggr_by == "last":
            # first imputation has missing value in last position
            dfs.append(preds_by_step.aggregate(lambda x: x[0]))
        else:
            raise ValueError(
                "aggregate_by can only be one of "
                "['mean', 'central', 'smooth_central', 'last']"
            )
    if isinstance(aggregate_by, str):
        return dfs[0]
    return dfs
