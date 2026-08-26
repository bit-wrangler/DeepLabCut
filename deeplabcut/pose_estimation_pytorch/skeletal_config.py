#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Shared configuration for the skeletal-constraint mechanisms (LLL / THT).

Both the *data* side (``create_skeleton_dictionary``, which emits the reference
length for the SVL landmark pair) and the *runner* side (``_compute_scale_factor_svl``
and the skeletal losses, which look that reference up again) need to agree on
which two landmarks stand in for snout-vent length.  They read that pair from a
single model-config key through :func:`resolve_svl_landmarks`, so they cannot
disagree.

Background: the CSV trait ``svl`` is a physical snout-to-*vent* measurement, but
the historical default pair is ``('snout', 'tail1')`` and ``tail1`` is the first
*tail* landmark, which sits behind the vent.  Over the labelled frames,
``||snout-tail1|| / ||snout-spine6||`` has a median of ~1.17, so the mm->pixel
scale is ~17% too large; that loosens the LLL threshold and inflates the
X-ray-derived THT radii.  ``spine6`` (the pelvic hub) is a near-exact vent proxy.
Setting ``svl_landmarks: [snout, spine6]`` in the model config switches to it.

This module deliberately imports nothing from ``deeplabcut`` so that it can be
imported from both ``pose_estimation_pytorch.data`` and
``pose_estimation_pytorch.runners`` without any risk of a circular import.
"""

from __future__ import annotations

from collections.abc import Sequence as _SequenceABC
from typing import Any, Mapping, Sequence

# Model-config key holding the landmark pair used as the SVL (snout-vent length)
# proxy.  Lives in pytorch_config.yaml (not the project config.yaml) so it is
# settable through run_cv.py's ``train_overrides`` and therefore recorded in the
# ``override__*`` columns of every cv_results_*.csv.
SVL_LANDMARKS_KEY = "svl_landmarks"

# Historical hardcoded pair.  Must remain the default so every existing result
# reproduces byte-identically when the config key is absent.
DEFAULT_SVL_LANDMARKS: tuple[str, str] = ("snout", "tail1")


def resolve_svl_landmarks(
    cfg: Mapping[str, Any] | None,
    bodyparts: Sequence[str] | None = None,
    source: str = "model config",
) -> tuple[str, str]:
    """Resolve the landmark pair used as the snout-vent length (SVL) proxy.

    Args:
        cfg: Mapping to read ``svl_landmarks`` from (the model config). May be
            None, in which case the default pair is returned.
        bodyparts: The project's bodypart names. When given *and* the key was
            explicitly set, both landmarks are checked for membership.
        source: Human-readable name of ``cfg``, used in error messages.

    Returns:
        A ``(first, second)`` tuple of bodypart names.

    Raises:
        ValueError: If the key is present but is not a sequence of exactly two
            distinct bodypart-name strings, or names a bodypart the project does
            not have. Never falls back silently.

    Note:
        When the key is *absent* the default pair is returned **without**
        membership validation, which preserves the existing graceful behaviour
        for projects that simply do not have a ``tail1`` landmark (there the
        skeletal code paths already degrade to a no-op). Validation applies only
        to a pair the user explicitly asked for.

        The rest of the pipeline honours the same exemption, so the contract holds
        end to end: ``create_skeleton_dictionary`` membership-validates only a pair
        that differs from :data:`DEFAULT_SVL_LANDMARKS`, and
        ``TrainingRunner._check_svl_reference_available`` skips its check when the
        pair is not expressible in the project's bodyparts. Multi-animal DLC
        projects, whose ``bodyparts`` is the sentinel string ``"MULTI!"``, depend
        on this.
    """
    if cfg is None or SVL_LANDMARKS_KEY not in cfg:
        return DEFAULT_SVL_LANDMARKS

    value = cfg[SVL_LANDMARKS_KEY]
    return validate_svl_landmarks(value, bodyparts, source=source)


def validate_svl_landmarks(
    value: Any,
    bodyparts: Sequence[str] | None = None,
    source: str = "model config",
) -> tuple[str, str]:
    """Validate an explicitly configured SVL landmark pair.

    See :func:`resolve_svl_landmarks` for the contract; this is the half that
    applies once we know the user set the key.
    """
    # str is itself a Sequence, but "snout" is not a pair of landmarks
    if isinstance(value, str) or not isinstance(value, _SequenceABC):
        raise ValueError(
            f"{SVL_LANDMARKS_KEY} in the {source} must be a list of exactly two "
            f"bodypart names, e.g. [snout, spine6]; got {value!r}."
        )

    pair = list(value)
    if len(pair) != 2:
        raise ValueError(
            f"{SVL_LANDMARKS_KEY} in the {source} must name exactly two "
            f"bodyparts; got {len(pair)} ({value!r})."
        )

    if not all(isinstance(name, str) for name in pair):
        raise ValueError(
            f"{SVL_LANDMARKS_KEY} in the {source} must contain bodypart name "
            f"strings; got {value!r}."
        )

    if pair[0] == pair[1]:
        raise ValueError(
            f"{SVL_LANDMARKS_KEY} in the {source} must name two *distinct* "
            f"bodyparts; got {value!r}."
        )

    if bodyparts is not None:
        available = list(bodyparts)
        missing = [name for name in pair if name not in available]
        if missing:
            raise ValueError(
                f"{SVL_LANDMARKS_KEY} in the {source} names bodypart(s) "
                f"{missing} which are not in this project; got {value!r}. "
                f"Available bodyparts: {available}."
            )

    return pair[0], pair[1]
