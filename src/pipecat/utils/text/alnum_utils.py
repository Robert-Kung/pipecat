#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Alphanumeric utilities for deciding whether text has anything to speak."""

import re

_COMPLETE_MARKUP_RE = re.compile(r"<[^>]+>")
"""Matched '<...>' pairs in a complete text; a lone '<' is content (``"5 < 10"``)."""


def has_alnum(text: str) -> bool:
    """Return True if *text* holds anything alphanumeric once markup is stripped.

    Answers whether there is anything left to speak: whitespace, punctuation and
    the letters inside a tag don't count, so ``"<break/>"`` is empty by this
    measure.

    Args:
        text: A complete text, such as everything an LLM produced in one response.

    Returns:
        True if any alphanumeric character remains outside '<...>' markup.
    """
    return any(ch.isalnum() for ch in _COMPLETE_MARKUP_RE.sub("", text))
