"""chi-IBR (chi-square Iterated Best Response) per-epoch dataset construction.

Port of "Construction of D(q^t)" (Algorithm 1, lines 6-9) from
fairseq-dro-mnmt (fairseq/data/multilingual/sampled_multi_dataset.py's
`get_virtual_indices`), plus length-sorted, padding-aware batching mirroring
the stock `LengthBatchSampler` (espnet2/samplers/length_batch_sampler.py),
applied to the resampled (not static), pooled-across-languages set each
epoch -- matching the fairseq reference's own approach, where batches are
formed by a global length sort with no per-language grouping.

Reference: Zhou et al., "Distributionally Robust Multilingual Machine
Translation", EMNLP 2021.

`solve_q` computes the target per-language sampling distribution q each
epoch; everything else in this module (resample_per_language, build_epoch_batches,
etc.) takes q as a plain input, agnostic to how it was produced.
"""

import logging
import math
import zlib
from typing import Callable, Dict, List, Sequence

import numpy as np
from typeguard import typechecked

logger = logging.getLogger(__name__)


def _stable_seed(s: str) -> int:
    """Deterministic (cross-process) integer seed component for a string.

    Python's built-in hash() is randomized per-process for strings by
    default (PYTHONHASHSEED), so it must not be used for anything that needs
    to be reproducible across separate runs (e.g. resuming training).
    """
    return zlib.crc32(s.encode("utf-8"))


def chi_square_divergence(q: Dict[str, float], p: Dict[str, float]) -> float:
    """chi^2(q, p) = 0.5 * sum_g p_g * (q_g / p_g - 1) ** 2."""
    return 0.5 * sum(p[g] * (q[g] / p[g] - 1) ** 2 for g in p)


def _q_of_eta(
    eta: float,
    p_train: Dict[str, float],
    v: Dict[str, float],
    min_prob: float,
) -> Dict[str, float]:
    """Closed-form best response for a fixed threshold eta.

    q_g(eta) propto p_train_g * relu(v_g - eta), then floor q_g / p_train_g at
    min_prob (so no language is ever fully excluded) and renormalize.
    Direct port of fairseq-dro-mnmt's ChiSquareResampleLabelSmoothedCrossEntropyCriterion
    .update_mw()'s inner p(eta) (fairseq/criterions/chi_square_resample.py).
    """
    pp = {g: p_train[g] * max(v[g] - eta, 0.0) for g in p_train}
    total = sum(pp.values())
    if total <= 0:
        # eta >= max(v): every relu term is exactly zero, including the
        # argmax's own (v_argmax - eta = 0). The true limiting behavior as
        # eta -> max(v) from below is concentration onto the argmax set
        # (split evenly across ties), NOT a fallback to p_train -- p_train
        # would understate chi^2 at this boundary and break the bisection
        # bracket (it needs chi^2 to be at its max here, not zero).
        max_v = max(v.values())
        argmax_set = [g for g in v if v[g] == max_v]
        q = {g: (1.0 / len(argmax_set) if g in argmax_set else 0.0) for g in p_train}
    else:
        q = {g: pp[g] / total for g in p_train}
    cq = {g: max(q[g] / p_train[g], min_prob) for g in p_train}
    cq_total = sum(cq[g] * p_train[g] for g in p_train)
    # float(...): eta can carry a numpy scalar type in from callers (e.g. via
    # np.sqrt upstream), which would otherwise leak numpy.float64 into the
    # returned dict -- and torch.save/load (PyTorch >=2.6, weights_only=True
    # default) rejects unpickling numpy scalar types by default.
    return {g: float(cq[g] * p_train[g] / cq_total) for g in p_train}


def _bisect(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> float:
    """Find eta in [lo, hi] with f(eta) ~= 0, assuming f is increasing.

    Direct port of fairseq-dro-mnmt's bisection() (chi_square_resample.py):
    widen the bracket first if it doesn't already contain a sign change, then
    standard bisection.
    """
    f_lo, f_hi = f(lo), f(hi)
    while f_lo > 0 or f_hi < 0:
        length = hi - lo
        if f_lo > 0:
            hi = lo
            lo = lo - 2 * length
        if f_hi < 0:
            lo = hi
            hi = hi + 2 * length
        f_lo, f_hi = f(lo), f(hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) <= tol:
            return mid
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    logger.warning(
        "chi_ibr_resample: bisection did not converge within %d iterations, "
        "returning midpoint of final bracket",
        max_iter,
    )
    return 0.5 * (lo + hi)


@typechecked
def solve_q(
    p_train: Dict[str, float],
    v: Dict[str, float],
    rho: float,
    min_prob: float = 0.2,
) -> Dict[str, float]:
    """Solve q* = argmax_{q: chi_square(q, p_train) <= rho} sum_g q_g * v_g.

    Algorithm 1, line 17. Direct port of fairseq-dro-mnmt's
    ChiSquareResampleLabelSmoothedCrossEntropyCriterion.update_mw()
    (fairseq/criterions/chi_square_resample.py).

    Args:
        p_train: fixed natural per-language data proportions, summing to 1.
        v: per-language value being maximized against (L_hat_g - b_g in the
            paper -- e.g. validation CTC loss minus baseline; baselines are
            not yet wired in here, so callers may pass v = L_hat directly).
        rho: radius of the chi-square ball around p_train.
        min_prob: floor on q_g / p_train_g, so no language is ever fully
            excluded from sampling.

    Returns:
        q: the reweighted per-language sampling distribution, summing to 1.

    Note: unlike the fairseq reference, this does not special-case "every
    language already beats its baseline" (there, `update_mw` falls back to
    un-baselined losses in that situation) -- not yet relevant since no
    baseline subtraction happens upstream of this function yet.
    """
    groups = list(p_train.keys())
    if set(v.keys()) != set(groups):
        raise ValueError("v and p_train must have the same set of language keys")
    if abs(sum(p_train.values()) - 1.0) > 1e-6:
        raise ValueError("p_train must sum to 1")
    if rho <= 0:
        return dict(p_train)

    eta_max = max(v.values())
    if eta_max <= 0:
        eta_min = eta_max - 1.0
    else:
        eta_min = -(1.0 / (math.sqrt(2 * rho + 1) - 1)) * eta_max

    def bisection_target(eta: float) -> float:
        q = _q_of_eta(eta, p_train, v, min_prob)
        return chi_square_divergence(q, p_train) - rho

    if bisection_target(eta_max) < 0:
        # rho exceeds the maximum divergence achievable at all (the min_prob
        # floor caps how concentrated q can get) -- no eta reaches the
        # boundary exactly, so just return the most extreme feasible point
        # rather than letting the bracket search widen forever looking for
        # a sign change that can't occur.
        return _q_of_eta(eta_max, p_train, v, min_prob)

    eta_star = _bisect(bisection_target, eta_min, eta_max)
    return _q_of_eta(eta_star, p_train, v, min_prob)


@typechecked
def compute_virtual_size(
    group_sizes: Dict[str, int],
    q: Dict[str, float],
    max_scale_up: float = 1.5,
) -> int:
    """Total budget to draw for one epoch under distribution q.

    Unit-agnostic: group_sizes can be per-language utterance counts or (as
    resample_per_language now uses) per-language total durations -- whatever
    unit group_sizes is in, virtual_size comes out in that same unit.

    Port of fairseq-dro-mnmt's default_virtual_size_func
    (fairseq/data/multilingual/sampled_multi_dataset.py): anchor the virtual
    size to whichever language is largest in real data (that language's
    virtual size equals its real size), scale every other language's virtual
    size relative to it under q, then cap the total at
    `max_scale_up * (real total size)` so a single heavily-upweighted small
    language can't blow up the epoch size.
    """
    if set(group_sizes.keys()) != set(q.keys()):
        raise ValueError("group_sizes and q must have the same set of language keys")

    real_total = sum(group_sizes.values())
    largest_group = max(group_sizes, key=group_sizes.get)
    largest_q = q[largest_group]
    if largest_q <= 0:
        return real_total
    virtual_size = group_sizes[largest_group] / largest_q
    return int(min(virtual_size, max_scale_up * real_total))


def _sample_until_duration(
    pool: List[str],
    utt2shape: Dict[str, Sequence[int]],
    target_duration: float,
    rng: np.random.RandomState,
) -> List[str]:
    """Randomly draw utterances from pool, accumulating duration, until the
    target is reached.

    Consumes one full shuffled pass through the pool without repeats first
    (equivalent to "without replacement" up to the real pool's own total
    duration); only reshuffles and continues (allowing repeats) if the
    target still hasn't been reached after using every real utterance once.
    May overshoot the target by up to one utterance's duration (the loop
    stops as soon as the running total reaches or exceeds it).
    """
    selected: List[str] = []
    accumulated = 0
    shuffled = list(pool)
    rng.shuffle(shuffled)
    idx = 0
    while accumulated < target_duration:
        if idx >= len(shuffled):
            rng.shuffle(shuffled)
            idx = 0
        utt = shuffled[idx]
        selected.append(utt)
        accumulated += utt2shape[utt][0]
        idx += 1
    return selected


@typechecked
def resample_per_language(
    category2utt: Dict[str, List[str]],
    q: Dict[str, float],
    utt2shape: Dict[str, Sequence[int]],
    seed: int,
    epoch: int,
    max_scale_up: float = 1.5,
) -> Dict[str, List[str]]:
    """Algorithm 1 lines 7-8: draw a duration-share n(q)_i per language, ungrouped.

    virtual_size (the total pool *duration* for this epoch) is derived from
    category2utt's real per-language total durations and q via
    compute_virtual_size -- it is not an independent input, mirroring
    fairseq's design (there, `adjust_sampling` is always called with
    `virtual_size=None`, letting `setup_sampling` derive it).

    Sampling is duration-based, not utterance-count-based: for each language
    i, utterances are drawn (without replacement while the real pool lasts,
    then with replacement if needed) until the accumulated duration reaches
    virtual_size * q_i. This matters because average utterance length varies
    a lot across languages/corpora -- count-based sampling would silently
    reflect each language's segmentation granularity rather than how much
    real speech it actually has (e.g. two languages with equal total
    duration but very different utterance counts, because one corpus was
    segmented into much shorter clips than the other).

    Unlike `resample_dataset`, this does NOT flatten/shuffle across languages
    -- batching needs the per-language grouping intact.

    Returns:
        Per-language dict of resampled utt-ids, with accumulated duration
        approximately matching virtual_size * q_i (may overshoot slightly,
        see _sample_until_duration).
    """
    if set(q.keys()) != set(category2utt.keys()):
        raise ValueError("q and category2utt must have the same set of language keys")
    if abs(sum(q.values()) - 1.0) > 1e-6:
        raise ValueError("q must sum to 1")

    group_durations = {
        lang: sum(utt2shape[u][0] for u in pool) for lang, pool in category2utt.items()
    }
    virtual_size = compute_virtual_size(group_durations, q, max_scale_up)

    languages = sorted(category2utt.keys())
    q_summary = ", ".join(f"{lang}={q[lang]:.4f}" for lang in languages)
    logger.info("chi_ibr_resample: epoch=%d input q: %s", epoch, q_summary)

    resampled: Dict[str, List[str]] = {}
    durations: Dict[str, int] = {}
    for lang in languages:
        pool = category2utt[lang]
        target_duration = virtual_size * q[lang]
        if target_duration <= 0:
            resampled[lang] = []
            durations[lang] = 0
            continue
        rng = np.random.RandomState([_stable_seed(lang), seed % (2**32), epoch])
        if group_durations[lang] < target_duration:
            logger.info(
                "chi_ibr_resample: language=%s requested duration=%d > pool duration=%d, "
                "sampling with replacement",
                lang,
                int(target_duration),
                group_durations[lang],
            )
        selected = _sample_until_duration(pool, utt2shape, target_duration, rng)
        resampled[lang] = selected
        durations[lang] = sum(utt2shape[u][0] for u in selected)

    total_duration = sum(durations.values())
    total_utts = sum(len(resampled[lang]) for lang in languages)
    summary = ", ".join(
        f"{lang}: duration={durations[lang]} "
        f"({100 * durations[lang] / total_duration:.1f}%), "
        f"utts={len(resampled[lang])} "
        f"({100 * len(resampled[lang]) / total_utts:.1f}%)"
        for lang in languages
    )
    logger.info(
        "chi_ibr_resample: epoch=%d virtual_size(duration)=%d pool composition: %s",
        epoch,
        virtual_size,
        summary,
    )
    return resampled


@typechecked
def resample_dataset(
    category2utt: Dict[str, List[str]],
    q: Dict[str, float],
    utt2shape: Dict[str, Sequence[int]],
    seed: int,
    epoch: int,
    max_scale_up: float = 1.5,
) -> List[str]:
    """Algorithm 1 lines 6-9 in full: resample per language, pool, shuffle.

    Convenience wrapper around resample_per_language for callers that just
    want the flat D(q^t) (e.g. matching the paper's literal pseudocode, or
    unit tests) rather than the per-language breakdown batching needs.

    Returns:
        A flat, shuffled list of utt-ids with total duration approximately
        matching virtual_size (see resample_per_language).
    """
    per_language = resample_per_language(
        category2utt, q, utt2shape, seed, epoch, max_scale_up
    )
    pooled = [utt for lang in sorted(per_language) for utt in per_language[lang]]
    shuffle_rng = np.random.RandomState([seed % (2**32), epoch])
    shuffle_rng.shuffle(pooled)
    return pooled


@typechecked
def pack_length_batches(
    utt_ids: List[str],
    utt2shape: Dict[str, Sequence[int]],
    batch_bins: int,
    min_batch_size: int = 1,
    sort_in_batch: str = "descending",
) -> List[List[str]]:
    """Length-sorted, padding-aware batching -- direct port of the stock
    espnet2.samplers.length_batch_sampler.LengthBatchSampler algorithm
    (espnet2/samplers/length_batch_sampler.py), applied to an arbitrary
    (e.g. resampled, pooled-across-languages) list of utt-ids.

    Sorts all utterances by length ascending (globally -- no per-language
    grouping), then walks the sorted list accumulating a batch, tracking
    bins = batch_count * max_length_in_batch (the actual padded-tensor size,
    since ascending order means the utterance just added is always the
    longest seen so far in the current batch). Once bins exceeds batch_bins
    (and the batch has reached min_batch_size), the batch is closed and a
    new one starts. This is deliberately different from a raw duration-sum
    bucket (pack_duration_batches's approach): it reflects what the padded
    batch tensor will actually cost, not the sum of unpadded utterance
    lengths.
    """
    keys = sorted(utt_ids, key=lambda k: utt2shape[k][0])

    batch_sizes: List[int] = []
    current_batch_keys: List[str] = []
    for key in keys:
        current_batch_keys.append(key)
        bins = len(current_batch_keys) * utt2shape[key][0]
        if bins > batch_bins and len(current_batch_keys) >= min_batch_size:
            batch_sizes.append(len(current_batch_keys))
            current_batch_keys = []
    if len(current_batch_keys) != 0:
        batch_sizes.append(len(current_batch_keys))

    if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
        for i in range(batch_sizes.pop(-1)):
            batch_sizes[-(i % len(batch_sizes)) - 1] += 1

    batches: List[List[str]] = []
    idx = 0
    for bs in batch_sizes:
        minibatch_keys = keys[idx : idx + bs]
        if sort_in_batch == "descending":
            minibatch_keys = list(reversed(minibatch_keys))
        batches.append(minibatch_keys)
        idx += bs
    return batches


@typechecked
def build_epoch_batches(
    category2utt: Dict[str, List[str]],
    q: Dict[str, float],
    utt2shape: Dict[str, Sequence[int]],
    batch_bins: int,
    seed: int,
    epoch: int,
    max_scale_up: float = 1.5,
    min_batch_size: int = 1,
) -> List[List[str]]:
    """Full pipeline: resample per language under q, pool all languages
    together, length-sort-and-batch the pooled set (pack_length_batches),
    shuffle batch order.

    This matches the fairseq reference's approach: no per-language grouping
    constraint -- a single batch can freely mix languages, as long as the
    utterances in it fit the shared (padding-aware) batch_bins budget. This
    is deliberately different from DurationLanguageBatchSampler
    (batch_type="duration_language"), which forces every batch to be
    single-language; chi-IBR does not use that constraint.

    This is the top-level entry point mirroring what
    DurationLanguageBatchSampler.__init__ does once, statically, for the
    whole run -- except recomputed fresh every epoch from a fresh q, and
    without the language-purity restriction.
    """
    resampled = resample_per_language(
        category2utt, q, utt2shape, seed, epoch, max_scale_up
    )
    pooled_utts = [utt for lang in sorted(resampled) for utt in resampled[lang]]
    batches = pack_length_batches(pooled_utts, utt2shape, batch_bins, min_batch_size)

    shuffle_rng = np.random.RandomState([seed % (2**32), epoch])
    perm = shuffle_rng.permutation(len(batches))
    return [batches[i] for i in perm]
