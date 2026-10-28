#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Alphanumeric utilities for deciding whether text has anything to speak."""

import re

from pipecat.utils.text.xml_function_tag_filter import FUNCTION_TAG_FLAGS, FUNCTION_TAG_PATTERNS

_FUNCTION_TAG_RES = tuple(re.compile(p, FUNCTION_TAG_FLAGS) for p in FUNCTION_TAG_PATTERNS)
"""Function-tag markup that XMLFunctionTagFilter removes whole, arguments included."""

_COMPLETE_MARKUP_RE = re.compile(r"<[^>]+>")
"""Matched '<...>' pairs in a complete text; a lone '<' is content (``"5 < 10"``)."""


def has_alnum(text: str) -> bool:
    """Return True if *text* holds anything alphanumeric once markup is stripped.

    Answers whether there is anything left to speak: whitespace, punctuation and
    the letters inside a tag don't count, so ``"<break/>"`` is empty by this
    measure. Function-tag markup (``<function=...>...</function>``) is removed
    whole, arguments included, the same way XMLFunctionTagFilter removes it
    before TTS. It judges the whole response at once, which approximates the
    filter: the filter runs on each sentence the TTS aggregates, so a tag split
    across sentences is matched here but not there.

    Args:
        text: A complete text, such as everything an LLM produced in one response.

    Returns:
        True if any alphanumeric character remains outside function tags and
        '<...>' markup.
    """
    for pattern in _FUNCTION_TAG_RES:
        text = pattern.sub("", text)
    return any(ch.isalnum() for ch in _COMPLETE_MARKUP_RE.sub("", text))
