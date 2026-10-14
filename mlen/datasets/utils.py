import itertools
import logging

import numpy as np
import tqdm

from .audio_reader import AudioReader
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#######################################################################
# Audio Loading functions from MERT: https://github.com/yizhilll/MERT
#######################################################################

def load_audio(tracks, max_keep, min_keep):
    n_long, n_short = 0, 0
    valid_tracks = []
    for ind, track in enumerate(tracks):
        sz = track['size']  
        if min_keep is not None and sz < min_keep:
            n_short += 1
        elif max_keep is not None and sz > max_keep:
            n_long += 1
        else:
            track['idx'] = ind
            valid_tracks.append(track)
    sizes = [track['size'] for track in valid_tracks]
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(valid_tracks)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return valid_tracks


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = []
        for line in tqdm.tqdm(f):
            labels.append(line.rstrip())
        # labels = [line.rstrip() ]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_numpy_label(label_path, inds, tot):
    labels = np.load(label_path, mmap_mode='r')
    assert (labels.shape[0] == tot), f"number of labels does not match ({labels.shape[0]} != {tot})"
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
    label_codebook_count=1,
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = []
        for line in tqdm.tqdm(f):
            lengths.append(len(line.rstrip().split()) / label_codebook_count)
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


def get_audio_file(file_path: str) -> tuple[np.ndarray, int]:
    """Compatibility helper; retain an AudioReader for repeated reads."""
    return AudioReader().read(file_path)


def read_from_s3(file_path: str) -> tuple[np.ndarray, int]:
    """Compatibility helper for standalone reads, including notebooks."""
    return AudioReader(location="S3").read(file_path)
