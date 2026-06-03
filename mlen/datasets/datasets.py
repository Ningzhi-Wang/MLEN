import logging
import os
import numpy as np
import torch
import torchaudio

from torch.utils.data import Dataset
from torch.nn import functional as F

from .utils import load_audio, load_label_offset, verify_label_lengths, get_audio_file
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
        ):
        meta_data = torch.load(meta_path)

        self.audio_root = root_path
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

        self.num_labels = len(label_paths)
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
        logger.info(
            f"crop_and_pad_to_maximum={self.crop_and_pad_to_maximum}, "
            f"max_sample_size={self.max_sample_size}"
        )

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, index):
        item = None
        while item is None:
            try:
                track = self.valid_tracks[index]
                aspect_idx = torch.randint(0, len(track['aspect_indices']), (1,)).item()
                wav = self.get_audio(track['track_name'])
                labels = self.get_labels(index)
                aspect_cond = self.aspect_embedding_table[track['aspect_indices'][aspect_idx]]
                aspect_target = self.text_embedding_table[track['text_indices'][aspect_idx]]
                item = {"id": index, "source": wav, "label_list": labels, "aspect_cond": aspect_cond, "aspect_target": aspect_target}
            except Exception as e:
                print(f'skip damaged data {index}')
                index = np.random.randint(len(self.sizes))
                
        return item

    def get_audio(self, track_name):
        wav_path = os.path.join(self.audio_root, track_name)

        wav, cur_sample_rate = get_audio_file(wav_path)
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

