import logging
from typing import Iterator, Optional, Tuple, Union, List

from typeguard import typechecked

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler
import random

random.seed(42)


class DurationBatchSampler(AbsSampler):
    """BatchSampler.

    Creates duration-equalized batches with mixed languages.
    Uses a greedy bin-packing approach over all utterances regardless
    of language/category, so each batch may contain multiple languages.
    """

    @typechecked
    def __init__(
        self,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        drop_last: bool = True,
        utt2category_file: Optional[str] = None,
        duration_batch_length: int = -1,
    ):
        assert batch_size > 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shape_files = shape_files

        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )

        # Sort all utterances by duration (descending) for better bin packing
        all_utts = sorted(
            first_utt2shape.keys(),
            key=lambda k: first_utt2shape[k][0],
            reverse=True,
        )

        total_frames = sum(first_utt2shape[k][0] for k in all_utts)
        avg_frames = total_frames / len(all_utts)

        if duration_batch_length != -1:
            target_dur = duration_batch_length
        else:
            target_dur = avg_frames * batch_size
        print(f"Target Duration {target_dur}")

        # Greedy bin packing over all utterances
        batches = [[]]
        bucket_sizes = [0]

        for utt in all_utts:
            length = first_utt2shape[utt][0]
            placed = False
            for i in range(len(bucket_sizes)):
                if bucket_sizes[i] + length <= target_dur:
                    batches[i].append(utt)
                    bucket_sizes[i] += length
                    placed = True
                    break
            if not placed:
                batches.append([utt])
                bucket_sizes.append(length)

        if drop_last:
            # Drop the last (potentially undersized) bin
            batches = [b for b in batches if len(b) > 1]

        random.shuffle(batches)
        self.batch_list = batches

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f")"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
