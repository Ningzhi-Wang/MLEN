import logging
import os
import numpy as np
import torch
import torchaudio

from torch.utils.data import Dataset
from torch.nn import functional as F

from .audio_reader import AudioReader
from .utils import load_audio, load_label_offset, verify_label_lengths
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AudioTextRVQDataset(Dataset):
    def __init__(
            self, 
            root_path, 
            meta_path, 
            label_paths=[],
            sample_rate=24000, 
            max_sample_size=24000*30, 
            min_sample_size=24000*5,
            crop_and_pad_to_maximum=True,
            shuffle=True,
            label_rates=25.,
            label_codebook_counts=1,
            normalize=False,
            label_processors=None,
            warn_on_label_length_mismatch=True,
            aspect_dropout=0.0,
        ):
        if not 0.0 <= aspect_dropout <= 1.0:
            raise ValueError("aspect_dropout must be in the range [0.0, 1.0]")
        meta_data = torch.load(meta_path)

        self.audio_root = root_path
        self.audio_reader = AudioReader()
        self.valid_tracks = load_audio(meta_data['tracks'], None, min_sample_size)
        inds = [track['idx'] for track in self.valid_tracks]
        self.sizes = [track['size'] for track in self.valid_tracks]
        tot = len(meta_data['tracks'])
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.crop_and_pad_to_maximum = crop_and_pad_to_maximum
        self.max_sample_size = max_sample_size
        self.text_embedding_table = meta_data['text_embedding_table']
        self.aspect_embedding_table = meta_data['aspect_eye']
        self.num_aspects = meta_data['num_aspects']
        self.normalize = normalize
        self.label_processors = label_processors
        # Probability of replacing a sample's aspect by the unconditional
        # (all-zero) condition; see get_item.
        self.aspect_dropout = aspect_dropout

        self.num_labels = len(label_paths)
        if self.num_labels > 0:
            self.label_prepared = True
            self.label_rates = (
                [label_rates for _ in range(len(label_paths))]
                if isinstance(label_rates, (float, int))
                else label_rates
            )
            self.label_codebook_counts = (
                [label_codebook_counts for _ in range(len(label_paths))]
                if isinstance(label_codebook_counts, int)
                else label_codebook_counts
            )
            logger.info('preparing labels')
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
            if warn_on_label_length_mismatch:
                for label_path, label_rate, label_codebook_count in zip(
                    label_paths, self.label_rates, self.label_codebook_counts
                ):
                    verify_label_lengths(
                        self.sizes,
                        self.sample_rate,
                        label_path,
                        label_rate,
                        inds,
                        tot,
                        label_codebook_count=label_codebook_count,
                    )
        else:
            self.label_prepared = False
        logger.info(
            f"crop_and_pad_to_maximum={self.crop_and_pad_to_maximum}, "
            f"max_sample_size={self.max_sample_size}"
        )

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, index):
        return self.get_item(index, aspect_dropout=self.aspect_dropout)

    def get_item(self, index, aspect_dropout=0.0):
        """Load one track with a random aspect.

        With probability ``aspect_dropout`` the aspect is dropped: the
        condition and target become all zeros and ``text_index`` is -1.
        ``condition_to_aspect_ids`` maps the zero condition to aspect -1,
        which losses such as whitened_or_rvq_loss treat as the unconditional
        (RVQ-only) sample and which per-aspect logging and label weights skip.
        """
        item = None
        while item is None:
            try:
                track = self.valid_tracks[index]
                # aspect_idx = np.random.choice([0, 1, 1, 2, 3, 4, 5, 6, 7])
                aspect_idx = torch.randint(0, len(track['aspect_indices']), (1,)).item()
                # positions = (track["aspect_indices"] == 1).nonzero().flatten()
                # aspect_idx = positions[0].item()

                wav = self.get_audio(track['track_name'])
                aspect_cond = self.aspect_embedding_table[track['aspect_indices'][aspect_idx]]
                aspect_target = self.text_embedding_table[track['text_indices'][aspect_idx]]
                # Index into text_embedding_table; used to look up per-label loss weights.
                text_index = int(track['text_indices'][aspect_idx])
                if aspect_dropout > 0.0 and torch.rand(()).item() < aspect_dropout:
                    aspect_cond = torch.zeros_like(aspect_cond)
                    aspect_target = torch.zeros_like(aspect_target)
                    text_index = -1

                # aspect_class_id = track['aspect_class_ids'][track['aspect_indices'][aspect_idx]]
                item = {
                    "id": index,
                    "source": wav,
                    "aspect_cond": aspect_cond,
                    "aspect_target": aspect_target,
                    "text_index": text_index,
                #    "aspect_class_id": aspect_class_id
                }
                if self.label_prepared:
                    item["label_list"] = self.get_labels(index)
            except Exception as e:
                # raise e
                # print(f'skip damaged data {index}')
                index = np.random.randint(len(self.sizes))
                
        return item

    def get_audio(self, track_name):
        wav_path = os.path.join(self.audio_root, track_name)

        wav, cur_sample_rate = self.audio_reader.read(wav_path)
        wav = torch.from_numpy(wav).float()
        if self.crop_and_pad_to_maximum:
            if wav.shape[0] > self.max_sample_size:
                start = torch.randint(0, wav.shape[0] - self.max_sample_size + 1, (1,)).item()
                wav = wav[start:start+self.max_sample_size]
            elif wav.shape[0] < self.max_sample_size:
                pad_size = self.max_sample_size - wav.shape[0]
                wav = F.pad(wav, (0, pad_size))
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_label(self, index, label_idx):
        with open(self.label_paths[label_idx]) as f:
            offset_s, offset_e = self.label_offsets_list[label_idx][index]
            f.seek(offset_s)
            label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav


class AspectDropoutView(Dataset):
    """The same tracks as ``dataset`` with a different aspect dropout rate.

    Lets a validation split carved out of the training dataset read the same
    metadata and labels without the training-time aspect dropout.
    """

    def __init__(self, dataset: AudioTextRVQDataset, aspect_dropout: float = 0.0):
        if not 0.0 <= aspect_dropout <= 1.0:
            raise ValueError("aspect_dropout must be in the range [0.0, 1.0]")
        self.dataset = dataset
        self.aspect_dropout = aspect_dropout

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.get_item(index, aspect_dropout=self.aspect_dropout)
