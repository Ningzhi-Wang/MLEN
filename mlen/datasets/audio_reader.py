"""Audio decoding with an S3 connection owned by the reading process."""

import io
import os

import boto3
import numpy as np
import soundfile as sf


class AudioReader:
    """Reuse a lazy S3 client within a dataset worker, never across processes.

    Construction stores configuration only. Pickling drops the client for
    spawned workers; the PID guard handles workers created with fork.
    """

    def __init__(
        self,
        location: str | None = None,
        endpoint_url: str = "https://ceph-private-object-rgw.comp-research.qmul.ac.uk",
        bucket: str = "c4dm-01",
    ):
        self.location = (
            location if location is not None else os.getenv("DATASET_LOCATION", "LOCAL")
        )
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self._client = None
        self._client_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        state["_client_pid"] = None
        return state

    def _get_client(self):
        pid = os.getpid()
        if self._client is None or self._client_pid != pid:
            aws_key = os.getenv("AWS_ACCESS_KEY")
            aws_secret = os.getenv("AWS_SECRET_KEY")
            assert aws_key is not None, "AWS_ACCESS_KEY not set"
            assert aws_secret is not None, "AWS_SECRET_KEY not set"
            session = boto3.session.Session()
            self._client = session.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
            )
            self._client_pid = pid
        return self._client

    def read(self, file_path: str) -> tuple[np.ndarray, int]:
        if self.location != "S3":
            return sf.read(file_path)

        client = self._get_client()
        with io.BytesIO() as file_buffer:
            client.download_fileobj(self.bucket, file_path, file_buffer)
            file_buffer.seek(0)
            return sf.read(file_buffer)
