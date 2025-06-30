from __future__ import annotations

import functools
import os
import pickle
from abc import ABC
from abc import abstractmethod
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO
from typing import Iterable
from typing import Iterator

import regex as re


class Tokenizer(ABC):
    vocab_size: int
    special_tokens: list[str]

    def __init__(self, vocab_size: int, special_tokens: list[str]) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    @abstractmethod
    def train_on_corpus(self, corpus_path: str | os.PathLike) -> None:
        pass

    def save_to_file(self, path: str | os.PathLike) -> None:
        with open(path, "wb") as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load_from_file(path: str | os.PathLike) -> Tokenizer:
        with open(path, "rb") as fd:
            tokenizer = pickle.load(fd)
        assert isinstance(tokenizer, Tokenizer)
        return tokenizer

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator:
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        pass


class BPETokenizer(Tokenizer):
    pre_tokenization_regex: str
    vocabulary: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    def __init__(
        self, vocab_size: int, special_tokens: list[str], pre_tokenization_regex: str
    ) -> None:
        super().__init__(vocab_size, special_tokens)
        self.pre_tokenization_regex = pre_tokenization_regex
        self.vocabulary = {}
        self.merges = []

    def train_on_corpus(
        self,
        corpus_path: str | os.PathLike,
        num_processes: int = 32,
        pre_tokenization_split_token: str | None = None,
    ) -> None:
        with open(corpus_path, "rb") as fd:
            chunk_boundaries = self._find_chunk_boundaries(
                fd, num_processes, pre_tokenization_split_token
            )

        pre_token_counts: Counter[bytes] = Counter()
        with Pool(num_processes) as pool:
            pre_tokenize = functools.partial(
                self._pre_tokenize_chunk,
                input_path=corpus_path,
                pre_tokenizer_regex=self.pre_tokenization_regex.encode("utf-8"),
                special_tokens=self.special_tokens,
            )
            for chunk_counts in pool.map(
                pre_tokenize,
                zip(chunk_boundaries[:-1], chunk_boundaries[1:]),
            ):
                pre_token_counts += chunk_counts

        # Initial vocabulary: all 256 bytes plus special tokens
        self.vocabulary = {}
        for i in range(256):
            self.vocabulary[i] = i.to_bytes()
        for special_token in self.special_tokens:
            self.vocabulary[len(self.vocabulary)] = special_token.encode("utf-8")

        self.merges = []
        freq_table = {
            tuple(byte.to_bytes() for byte in k): v for k, v in pre_token_counts.items()
        }
        byte_pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for pre_token, count in freq_table.items():
            for byte_pair in zip(pre_token[:-1], pre_token[1:]):
                byte_pair_counts[byte_pair] = byte_pair_counts.get(byte_pair, 0) + count
        for i in range(self.vocab_size - len(self.vocabulary)):
            # Break ties deterministically by taking lexicographically greatest byte pair
            most_common_pair = max(
                byte_pair_counts, key=lambda pair: (byte_pair_counts[pair], pair)
            )
            self.merges.append(most_common_pair)
            new_token = bytes().join(most_common_pair)
            self.vocabulary[len(self.vocabulary)] = new_token
            byte_pair_counts[most_common_pair] = 0

            # Substitute pair in frequency counts by new token
            def merge(
                pre_token: tuple[bytes, ...], count: int, new_pair: tuple[bytes, bytes]
            ) -> tuple[tuple[bytes, ...], int]:
                nonlocal byte_pair_counts
                new_pre_token = list(pre_token)
                i, j = 0, 0
                while i < len(new_pre_token) - 1:
                    if (new_pre_token[i], new_pre_token[i + 1]) == new_pair:
                        new_token = bytes().join(new_pair)
                        new_pre_token[i] = new_token
                        del new_pre_token[i + 1]
                        if j > 0:
                            byte_pair_counts[(pre_token[j - 1], new_pair[0])] -= count
                            byte_pair_counts[(pre_token[j - 1], new_token)] += count
                        if j < len(pre_token) - 2:
                            byte_pair_counts[(new_pair[1], pre_token[j + 2])] -= count
                            byte_pair_counts[(new_token, pre_token[j + 2])] += count
                        j += 1
                    i += 1
                    j += 1
                return tuple(new_pre_token), count

            freq_table = dict(
                merge(k, v, most_common_pair) for k, v in freq_table.items()
            )

    @staticmethod
    def _find_chunk_boundaries(
        file: BinaryIO, desired_num_chunks: int, split_special_token: str | None
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        # Function adapted from
        # https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                if split_special_token is not None:
                    found_at = mini_chunk.find(split_special_token.encode("utf-8"))
                    if found_at != -1:
                        chunk_boundaries[bi] = initial_position + found_at
                        break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    @staticmethod
    def _pre_tokenize_chunk(
        chunk_limits: tuple[int, int],
        input_path: str | os.PathLike,
        pre_tokenizer_regex: bytes,
        special_tokens: list[str],
    ) -> dict[bytes, int]:
        start, end = chunk_limits
        chunk_size = end - start
        with open(input_path, "rb") as fd:
            fd.seek(start)
            chunk = fd.read(chunk_size)
        # Split on special tokens to avoid merging across document boundaries
        special_tokens_regex = "|".join(
            re.escape(token) for token in special_tokens
        ).encode("utf-8")
        counts: dict[bytes, int] = {}
        for chunk_doc in re.split(special_tokens_regex, chunk):
            for match in re.finditer(pre_tokenizer_regex, chunk_doc):
                pre_token = match.group()
                counts[pre_token] = counts.get(pre_token, 0) + 1
        return counts

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError
