"""chi-IBR epoch-aware iterator factory.

Unlike SequenceIterFactory, which holds a fixed batch list built once for the
whole run and only reorders/windows it per epoch, this factory rebuilds batch
*composition* fresh every epoch from the current target distribution q (see
espnet2.samplers.chi_ibr_resample).

Mirrors SequenceIterFactory's interface and DataLoader-construction so it can
be used as a drop-in AbsIterFactory (espnet2/iterators/sequence_iter_factory.py).
"""

import logging
from functools import partial
from typing import Dict, List, Optional, Sequence

from torch.utils.data import DataLoader
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.sequence_iter_factory import worker_init_fn
from espnet2.samplers.chi_ibr_resample import build_epoch_batches

logger = logging.getLogger(__name__)


class EpochAwareSequenceIterFactory(AbsIterFactory):
    """Builds a fresh, chi-IBR-resampled DataLoader every epoch.

    q defaults to p_train (the real per-language data proportions) until
    set_q() is called -- matching Algorithm 1's q^0 = p_train for the initial
    epoch(s) before any weight-update mechanism has produced a real q.
    """

    @typechecked
    def __init__(
        self,
        dataset,
        category2utt: Dict[str, List[str]],
        utt2shape: Dict[str, Sequence[int]],
        batch_bins: int,
        seed: int = 0,
        max_scale_up: float = 1.5,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
    ):
        self.dataset = dataset
        self.category2utt = category2utt
        self.utt2shape = utt2shape
        self.batch_bins = batch_bins
        self.seed = seed
        self.max_scale_up = max_scale_up
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory

        # Trainer.run reads .groups (list of unique group ids) to seed
        # group_dro_weights.pth uniformly on a fresh (non-resumed) start --
        # same contract as SequenceIterFactory.groups / batch_sampler.groups
        # (set from the categories found in utt2category).
        self.groups: List[str] = list(category2utt.keys())

        # Duration-based, matching abs_task.py's chi_ibr_p_train.pth (must
        # stay consistent with what solve_q uses as p_train).
        group_durations = {
            lang: sum(utt2shape[u][0] for u in utts)
            for lang, utts in category2utt.items()
        }
        real_total = sum(group_durations.values())
        self.q: Dict[str, float] = {
            lang: dur / real_total for lang, dur in group_durations.items()
        }
        logger.info("EpochAwareSequenceIterFactory: initial q = p_train: %s", self.q)

    def set_q(self, q: Dict[str, float]) -> None:
        if set(q.keys()) != set(self.category2utt.keys()):
            raise ValueError(
                f"set_q received languages {sorted(q.keys())} but "
                f"category2utt has {sorted(self.category2utt.keys())}"
            )
        self.q = dict(q)

    def build_iter(self, epoch: int, shuffle: Optional[bool] = None) -> DataLoader:
        batches = build_epoch_batches(
            self.category2utt,
            self.q,
            self.utt2shape,
            self.batch_bins,
            self.seed,
            epoch,
            self.max_scale_up,
        )

        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        return DataLoader(
            dataset=self.dataset,
            batch_sampler=batches,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=partial(worker_init_fn, base_seed=epoch + self.seed),
            **kwargs,
        )
