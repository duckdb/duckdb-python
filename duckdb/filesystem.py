from io import TextIOBase  # noqa: D100
from typing import IO

from fsspec import AbstractFileSystem
from fsspec.implementations.memory import MemoryFile, MemoryFileSystem

from .bytes_io_wrapper import BytesIOWrapper


def is_file_like(obj) -> bool:  # noqa: D103, ANN001
    # We only care that we can read from the file
    return hasattr(obj, "read") and hasattr(obj, "seek")


class ModifiedMemoryFileSystem(MemoryFileSystem):  # noqa: D101
    protocol = ("DUCKDB_INTERNAL_OBJECTSTORE",)
    # defer to the original implementation that doesn't hardcode the protocol
    _strip_protocol = classmethod(AbstractFileSystem._strip_protocol.__func__)

    def add_file(self, object: IO, path: str) -> None:  # noqa: D102
        if not is_file_like(object):
            msg = "Can not read from a non file-like object"
            raise ValueError(msg)
        path = self._strip_protocol(path)
        if isinstance(object, TextIOBase):
            # Wrap this so that we can return a bytes object from 'read'
            object = BytesIOWrapper(object)
        self.store[path] = MemoryFile(self, path, object.read())
