"""Read migrated targets without opening their legacy provenance .pt file.

The manifest owns file locations. Tracks are restored to source_row order so
legacy RVQ line numbers and seeded train/validation splits stay reproducible.
The compatibility metadata view keeps existing dataset/analysis callers small.

Training automatically selects META_PATH/targets when present, or accepts
--targets-path DIRECTORY (also manifest.json). RVQ text files and target_geometry.pt
remain under META_PATH. PyArrow is required for the exported Parquet tables.
"""

import json
from pathlib import Path

import numpy as np
import torch


def is_target_store(path):
    path = Path(path)
    return path.is_dir() or path.name == "manifest.json"


class TargetStore:
    def __init__(self, path):
        path = Path(path)
        if not path.is_dir() and not (path.name == "manifest.json" and path.is_file()):
            raise FileNotFoundError(f"Expected a target directory or manifest.json: {path}")
        self.root = (path if path.is_dir() else path.parent).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported target-store schema_version")
        self.aspect = self.manifest["targets"]["aspects"]
        if self.aspect.get("version") != 1 or self.aspect.get("scope") != "track":
            raise ValueError("Unsupported aspect target version or scope")
        self.counts = self.manifest["counts"]
        self.num_aspects = self.counts["aspects"]
        self.embedding_dim = self.aspect["embedding_dim"]

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Target-store path escapes the dataset: {relative}")
        return path

    def array(self, relative, shape=None, dtype=None):
        # Copy-on-write mapping gives Torch writable storage without modifying
        # the exported array or loading the entire embedding table eagerly.
        value = np.load(self.path(relative), mmap_mode="c", allow_pickle=False)
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"Unexpected shape for {relative}: {value.shape}, expected {shape}")
        if dtype is not None and value.dtype != np.dtype(dtype):
            raise ValueError(f"Unexpected dtype for {relative}: {value.dtype}, expected {dtype}")
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"Non-finite or nonnumeric target array: {relative}")
        return torch.from_numpy(value)

    def load_metadata(self):
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        tracks = pq.read_table(self.path(self.manifest["tracks"]))
        n = len(tracks)
        if n != self.counts["tracks"]:
            raise ValueError("Track count differs from the manifest")
        if not np.array_equal(tracks["source_row"].to_numpy(), np.arange(n)):
            tracks = tracks.take(pc.sort_indices(tracks, sort_keys=[("source_row", "ascending")]))
        if not np.array_equal(tracks["source_row"].to_numpy(), np.arange(n)):
            raise ValueError("source_row must contain each legacy row exactly once")
        track_ids = tracks["track_id"]
        if track_ids.null_count or len(pc.unique(track_ids)) != n:
            raise ValueError("track_id must be non-null and unique")

        labels = pq.read_table(self.path(self.aspect["index"]))
        if len(labels) != self.counts["aspect_labels"] or any(c.null_count for c in labels.columns):
            raise ValueError("Invalid aspect label count or null labels")
        rows = pc.index_in(labels["track_id"], value_set=track_ids)
        if rows.null_count:
            raise ValueError("Aspect labels reference an unknown track_id")
        rows = rows.to_numpy().astype(np.int64)
        aids = labels["aspect_id"].to_numpy()
        tids = labels["text_id"].to_numpy()
        if (aids.dtype.kind not in "iu" or tids.dtype.kind not in "iu" or
                ((aids < 0) | (aids >= self.num_aspects)).any() or
                ((tids < 0) | (tids >= self.counts["texts"])).any()):
            raise ValueError("Out-of-range aspect_id or text_id")
        # Dense pair counts are linear in label count and avoid building a
        # hash table of every track/aspect pair at each worker/rank startup.
        pair_counts = np.bincount(rows * self.num_aspects + aids,
                                  minlength=n * self.num_aspects)
        if (pair_counts > 1).any():
            raise ValueError("Duplicate track/aspect label")
        dense = np.full((n, self.num_aspects), -1, dtype=np.int64)
        dense[rows, aids] = tids
        # Prepare all ragged row slices together, rather than invoking NumPy
        # flatnonzero and advanced indexing separately for every track.
        present_rows, present_aspects = np.nonzero(dense >= 0)
        present_texts = dense[present_rows, present_aspects]
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(np.bincount(present_rows, minlength=n), out=offsets[1:])

        vocabulary = pq.read_table(self.path(self.aspect["text_vocabulary"]))
        vocabulary = vocabulary.take(pc.sort_indices(vocabulary, sort_keys=[("text_id", "ascending")]))
        if not np.array_equal(vocabulary["text_id"].to_numpy(), np.arange(self.counts["texts"])):
            raise ValueError("Text vocabulary IDs do not match the embedding rows")
        aspect_vocabulary = pq.read_table(self.path(self.aspect["vocabulary"]))
        aspect_vocabulary = aspect_vocabulary.take(pc.sort_indices(aspect_vocabulary, sort_keys=[("aspect_id", "ascending")]))
        if not np.array_equal(aspect_vocabulary["aspect_id"].to_numpy(), np.arange(self.num_aspects)):
            raise ValueError("Aspect vocabulary IDs do not match the aspect rows")

        records = tracks.to_pylist()
        for i, record in enumerate(records):
            raw_extras = record.pop("extras_json")
            if raw_extras:
                extras = json.loads(raw_extras)
                if not isinstance(extras, dict):
                    raise ValueError("Track extras_json must be an object")
                # Explicit exported fields take precedence over legacy extras.
                record.update({k: v for k, v in extras.items() if k not in record})
            record["size"] = record["num_samples"]
            start, end = offsets[i], offsets[i + 1]
            record["aspect_indices"] = present_aspects[start:end]
            record["text_indices"] = present_texts[start:end]
        return {
            "tracks": records,
            "num_aspects": self.num_aspects,
            "aspect_eye": self.array(self.aspect["aspect_embeddings"], (self.num_aspects, self.num_aspects)),
            "text_embedding_table": self.array(self.aspect["text_embeddings"],
                (self.counts["texts"], self.embedding_dim), self.aspect["embedding_dtype"]),
            "unique_texts": vocabulary["text"].to_pylist(),
            "aspect_names": aspect_vocabulary["name"].to_pylist(),
        }

    def load_statistics(self, section=None):
        record = self.manifest.get("target_statistics")
        if record is None:
            return {} if section is None else None
        if record.get("version") != 1:
            raise ValueError("Unsupported target statistics version")
        document = json.loads(self.path(record["metadata"]).read_text())
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported statistics schema_version")
        alignment = document["alignment"]
        expected_hash = self.manifest["files"][self.aspect["text_embeddings"]]["sha256"]
        if (alignment["text_embeddings_sha256"] != expected_hash or
                alignment["aspect_ids"] != list(range(self.num_aspects))):
            raise ValueError("Statistics do not match the exported embeddings/aspect IDs")

        def resolve(value):
            if isinstance(value, dict):
                if "array_path" in value:
                    return self.array(value["array_path"], value["shape"], value["dtype"])
                return {k: resolve(v) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve(v) for v in value]
            return value

        values = document["values"]
        result = resolve(values if section is None else values.get(section))
        selected = result if section is None else {section: result}
        whitening = selected.get("aspect_whitening")
        if whitening is not None:
            for key, shape in (("means", (self.num_aspects, self.embedding_dim)),
                               ("whitening_matrices", (self.num_aspects, self.embedding_dim, self.embedding_dim))):
                tensor = whitening.get(key)
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or not tensor.is_floating_point():
                    raise ValueError(f"Invalid whitening {key}; expected floating point {shape}")
        weights = selected.get("label_weights")
        if weights is not None:
            tensor = weights.get("weights")
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != (self.counts["texts"],)
                    or not tensor.is_floating_point() or (tensor <= 0).any()):
                raise ValueError("Invalid per-text label weights")
        return result


def load_metadata(path):
    """Read a target directory/manifest, or an explicitly selected legacy file."""
    if is_target_store(path):
        return TargetStore(path).load_metadata()
    return torch.load(path, map_location="cpu", weights_only=False)


def load_target_statistics(path, section=None):
    if is_target_store(path):
        return TargetStore(path).load_statistics(section)
    values = torch.load(path, map_location="cpu", weights_only=False).get("target_statistics", {})
    return values if section is None else values.get(section)


def resolve_training_metadata(meta_path, targets_path=None):
    """Prefer the migrated export; leave RVQ/geometry in the metadata directory."""
    meta_path = Path(meta_path)
    candidate = Path(targets_path) if targets_path is not None else meta_path / "targets"
    if targets_path is not None or candidate.exists():
        TargetStore(candidate)  # Fail early on an explicit/incomplete export.
        return candidate, candidate
    return meta_path / "track_metadata.pt", meta_path / "track_metadata_with_centroids.pt"
