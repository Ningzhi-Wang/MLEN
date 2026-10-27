from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

from .datasets import AudioTextRVQDataset


@dataclass
class AudioTextRVQCollator:
    max_sample_size: int = 24_000 * 30
    sample_rate: int = 24_000
    hop_length: int = 240
    n_fold: int = 4
    label_rates: float | Sequence[float] = 25.0
    label_codebook_counts: int | Sequence[int] = 1
    pad_list: int | Sequence[int] = 0
    pad_audio: bool = False
    fixed_sample_size: bool = False
    random_crop: bool = True
    single_target: bool = False
    return_net_input: bool = False

    @property
    def audio_unit(self) -> int:
        return self.hop_length * self.n_fold

    def _batch_audio_size(self, audios: Sequence[torch.Tensor]) -> int:
        audio_sizes = [audio.numel() for audio in audios]
        if self.fixed_sample_size:
            audio_size = self.max_sample_size
        elif self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)

        audio_size = (audio_size // self.audio_unit) * self.audio_unit
        if audio_size <= 0:
            audio_size = self.audio_unit
        return audio_size

    def _crop_or_pad_audio(self, audio: torch.Tensor, audio_size: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        audio = audio.flatten()
        diff = audio.numel() - audio_size
        padding_mask = torch.zeros(audio_size, dtype=torch.bool)

        if diff == 0:
            return audio, 0, padding_mask

        if diff < 0:
            padding_mask[audio.numel():] = True
            return F.pad(audio, (0, -diff)), 0, padding_mask
            # padded_audio = F.pad(audio, (0, -diff))
            # return padded_audio, 0, padding_mask

        start = 0
        if self.random_crop:
            max_start = (diff // self.audio_unit) * self.audio_unit
            start_unit = torch.randint(0, max_start // self.audio_unit + 1, (1,)).item()
            start = start_unit * self.audio_unit
        return audio[start:start + audio_size], start, padding_mask

    @staticmethod
    def _as_label_tensor(label) -> torch.Tensor:
        if isinstance(label, torch.Tensor):
            return label.long()
        if isinstance(label, str):
            values = label.strip().split()
            return torch.LongTensor([int(value) for value in values])
        return torch.as_tensor(label, dtype=torch.long)

    @staticmethod
    def _collate_tokens(targets: Sequence[torch.Tensor], pad: int) -> torch.Tensor:
        if len(targets) == 0:
            return torch.empty(0, dtype=torch.long)
        size = max(target.numel() for target in targets)
        res = targets[0].new_full((len(targets), size), pad)
        for i, target in enumerate(targets):
            res[i, :target.numel()] = target
        return res

    def _label_rates_for(self, num_labels: int) -> list[float]:
        if isinstance(self.label_rates, (float, int)):
            return [float(self.label_rates) for _ in range(num_labels)]
        return [float(rate) for rate in self.label_rates]

    def _pad_list_for(self, num_labels: int) -> list[int]:
        if isinstance(self.pad_list, int):
            return [self.pad_list for _ in range(num_labels)]
        return [int(pad) for pad in self.pad_list]

    def _codebook_counts_for(self, num_labels: int) -> list[int]:
        if isinstance(self.label_codebook_counts, int):
            return [self.label_codebook_counts for _ in range(num_labels)]
        return [int(count) for count in self.label_codebook_counts]

    def _targets_by_label(self, samples: Sequence[dict]) -> list[list[torch.Tensor]]:
        if "label_list" not in samples[0]:
            return []
        num_labels = len(samples[0]["label_list"])
        return [
            [self._as_label_tensor(sample["label_list"][label_idx]) for sample in samples]
            for label_idx in range(num_labels)
        ]

    @staticmethod
    def _collate_class_ids(samples):
        class_id_rows = []

        for sample in samples:
            value = sample["aspect_class_id"]

            if isinstance(value, torch.Tensor):
                ids = value.detach().long().flatten()
            else:
                ids = torch.as_tensor(value, dtype=torch.long).flatten()

            # Remove padding that might already exist in metadata.
            ids = ids[ids >= 0]

            if ids.numel() == 0:
                raise ValueError(
                    "The selected aspect has no valid class IDs"
                )

            # Avoid repeated positive classes.
            ids = torch.unique(ids)
            class_id_rows.append(ids)

        padded_ids = pad_sequence(
            class_id_rows,
            batch_first=True,
            padding_value=-1,
        )

        return padded_ids

    def _collate_frm_label(
        self,
        targets: Sequence[torch.Tensor],
        audio_size: int,
        audio_starts: Sequence[int],
        label_rate: float,
        pad: int,
        codebook_count: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if label_rate == self.sample_rate / self.audio_unit:
            frm_starts = [int(start // self.audio_unit) for start in audio_starts]
            frm_size = int(audio_size // self.audio_unit)
        else:
            s2f = label_rate / self.sample_rate
            frm_starts = [int(round(start * s2f)) for start in audio_starts]
            frm_size = int(round(audio_size * s2f))

        if codebook_count == 1:
            if not self.pad_audio:
                rem_size = [target.numel() - start for target, start in zip(targets, frm_starts)]
                frm_size = min(frm_size, *rem_size)
            targets = [target[start:start + frm_size] for target, start in zip(targets, frm_starts)]
        else:
            frame_lengths = [target.numel() // codebook_count for target in targets]
            if not self.pad_audio:
                rem_size = [length - start for length, start in zip(frame_lengths, frm_starts)]
                frm_size = min(frm_size, *rem_size)
            cropped = []
            for target, start in zip(targets, frm_starts):
                usable = (target.numel() // codebook_count) * codebook_count
                target = target[:usable].view(codebook_count, -1)
                cropped.append(target[:, start:start + frm_size].reshape(-1))
            targets = cropped

        lengths = torch.LongTensor([target.numel() for target in targets])
        ntokens = lengths.sum().item()
        targets = self._collate_tokens(targets, pad)
        return targets, lengths, ntokens

    def _collate_seq_label(
        self, targets: Sequence[torch.Tensor], pad: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        lengths = torch.LongTensor([target.numel() for target in targets])
        ntokens = lengths.sum().item()
        targets = self._collate_tokens(targets, pad)
        return targets, lengths, ntokens

    def _collate_label(
        self, targets_by_label: Sequence[Sequence[torch.Tensor]], audio_size: int, audio_starts: Sequence[int]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        targets_list, lengths_list, ntokens_list = [], [], []
        label_rates = self._label_rates_for(len(targets_by_label))
        pad_list = self._pad_list_for(len(targets_by_label))
        codebook_counts = self._codebook_counts_for(len(targets_by_label))

        for targets, label_rate, pad, codebook_count in zip(
            targets_by_label, label_rates, pad_list, codebook_counts
        ):
            if label_rate == -1.0:
                targets, lengths, ntokens = self._collate_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self._collate_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad, codebook_count
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def __call__(self, samples: Sequence[dict]) -> dict:
        samples = [sample for sample in samples if sample.get("source") is not None]
        if len(samples) == 0:
            return {}

        audios = [sample["source"] for sample in samples]
        audio_size = self._batch_audio_size(audios)

        collated = [
            self._crop_or_pad_audio(audio, audio_size)
            for audio in audios
        ]
        sources, starts, padding_masks = zip(*collated)
        targets_by_label = self._targets_by_label(samples)
        # aspect_class_ids = self._collate_class_ids(samples)

        batch = {
            "id": torch.LongTensor([sample["id"] for sample in samples]),
            "source": torch.stack(sources),
            "padding_mask": torch.stack(padding_masks),
            "audio_starts": torch.LongTensor(starts),
            # [B, A, D] targets and [B, A] label indices for every aspect; -1
            # marks an aspect the track lacks. The condition is sampled later.
            "aspect_target": torch.stack([sample["aspect_target"] for sample in samples]),
            "text_index": torch.stack([
                torch.as_tensor(sample["text_index"], dtype=torch.long) for sample in samples
            ]),
            # "aspect_class_id": aspect_class_ids,
            # "aspect_class_mask": aspect_class_ids.ne(-1),
        }

        if len(targets_by_label) > 0:
            target_list, target_lengths_list, ntokens_list = self._collate_label(
                targets_by_label, audio_size, starts
            )
            batch["target_list"] = target_list
            batch["target_lengths_list"] = target_lengths_list
            batch["ntokens_list"] = ntokens_list

            if len(target_list) == 1:
                batch["target"] = target_list[0]
                batch["target_lengths"] = target_lengths_list[0]
                batch["ntokens"] = ntokens_list[0]
            else:
                batch["target"] = torch.stack(target_list, dim=1)
                batch["target_lengths"] = torch.stack(target_lengths_list, dim=1)
                batch["ntokens"] = sum(ntokens_list)
        else:
            batch["target_list"] = None
            batch["target_lengths_list"] = None
            batch["ntokens_list"] = None
            batch["target"] = None
            batch["target_lengths"] = None
            batch["ntokens"] = None

        if self.return_net_input:
            batch["net_input"] = {
                "source": batch["source"],
                "padding_mask": batch["padding_mask"],
            }
            batch["net_input_target"] = batch["aspect_target"]

        return batch


class MLENDataModule(LightningDataModule):
    def __init__(
        self,
        root_path: str,
        train_meta_path: Optional[str] = None,
        val_meta_path: Optional[str] = None,
        test_meta_path: Optional[str] = None,
        train_label_paths: Optional[Iterable[str]] = None,
        val_label_paths: Optional[Iterable[str]] = None,
        test_label_paths: Optional[Iterable[str]] = None,
        batch_size: int = 8,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        drop_last: bool = True,
        sample_rate: int = 24000,
        max_sample_size: int = 24000 * 30,
        min_sample_size: int = 24000 * 5,
        hop_length: int = 240,
        label_rate: float | Sequence[float] = 25.0,
        label_codebook_counts: int | Sequence[int] = 1,
        pad_list: int | Sequence[int] = 0,
        pad_audio: bool = False,
        fixed_sample_size: bool = False,
        train_random_crop: bool = True,
        normalize: bool = False,
        single_target: bool = False,
        return_net_input: bool = False,
        warn_on_label_length_mismatch: bool = True,
        val_split_fraction: float = 0.0,
        val_split_seed: int = 0,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        if not 0.0 <= val_split_fraction < 1.0:
            raise ValueError("val_split_fraction must be in the range [0.0, 1.0).")

        self.root_path = root_path
        self.meta_paths = {
            "train": train_meta_path,
            "val": val_meta_path,
            "test": test_meta_path,
        }
        self.label_paths = {
            "train": list(train_label_paths or []),
            "val": list(val_label_paths or []),
            "test": list(test_label_paths or []),
        }

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.val_split_fraction = val_split_fraction
        self.val_split_seed = val_split_seed

        self.dataset_kwargs = {
            "sample_rate": sample_rate,
            "max_sample_size": max_sample_size,
            "min_sample_size": min_sample_size,
            "crop_and_pad_to_maximum": False,
            "label_rates": label_rate,
            "label_codebook_counts": label_codebook_counts,
            "normalize": normalize,
            "warn_on_label_length_mismatch": warn_on_label_length_mismatch,
        }

        n_fold_label_rate = label_rate[0] if isinstance(label_rate, Sequence) else label_rate
        n_fold = int(100 // n_fold_label_rate)
        self.train_collator = AudioTextRVQCollator(
            max_sample_size=max_sample_size,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fold=n_fold,
            label_rates=label_rate,
            label_codebook_counts=label_codebook_counts,
            pad_list=pad_list,
            pad_audio=pad_audio,
            fixed_sample_size=fixed_sample_size,
            random_crop=train_random_crop,
            single_target=single_target,
            return_net_input=return_net_input,
        )
        self.eval_collator = AudioTextRVQCollator(
            max_sample_size=max_sample_size,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fold=n_fold,
            label_rates=label_rate,
            label_codebook_counts=label_codebook_counts,
            pad_list=pad_list,
            pad_audio=pad_audio,
            fixed_sample_size=fixed_sample_size,
            random_crop=False,
            single_target=single_target,
            return_net_input=return_net_input,
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make_dataset(self, split: str) -> Optional[AudioTextRVQDataset]:
        meta_path = self.meta_paths[split]
        if meta_path is None:
            return None
        return AudioTextRVQDataset(
            root_path=self.root_path,
            meta_path=meta_path,
            label_paths=self.label_paths[split],
            **self.dataset_kwargs,
        )

    def _fit_datasets(self) -> tuple[Optional[Dataset], Optional[Dataset]]:
        train_dataset = self._make_dataset("train")
        if train_dataset is None:
            return None, None

        if self.meta_paths["val"] is not None:
            return train_dataset, self._make_dataset("val")

        if self.val_split_fraction == 0.0:
            return train_dataset, None

        val_size = int(len(train_dataset) * self.val_split_fraction)
        if val_size == 0:
            val_size = 1
        train_size = len(train_dataset) - val_size
        if train_size <= 0:
            raise ValueError(
                "val_split_fraction leaves no samples for training. "
                "Use a smaller fraction or provide an explicit val_meta_path."
            )

        generator = torch.Generator().manual_seed(self.val_split_seed)
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=generator,
        )
        return train_subset, val_subset

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset, self.val_dataset = self._fit_datasets()
        if stage in (None, "validate"):
            _, self.val_dataset = self._fit_datasets()
        if stage in (None, "test"):
            self.test_dataset = self._make_dataset("test")

    def _loader(
        self,
        dataset: Optional[Dataset],
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        collate_fn: AudioTextRVQCollator,
    ) -> Optional[DataLoader]:
        if dataset is None:
            return None
        worker_options = {}
        if self.num_workers > 0:
            worker_options = {
                "persistent_workers": (
                    True if self.persistent_workers is None else self.persistent_workers
                ),
                "prefetch_factor": self.prefetch_factor,
            }
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **worker_options,
        )

    def train_dataloader(self):
        return self._loader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            collate_fn=self.train_collator,
        )

    def val_dataloader(self):
        return self._loader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.eval_collator,
        )

    def test_dataloader(self):
        return self._loader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.eval_collator,
        )


MLENDatamodule = MLENDataModule
