#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for has_alnum(), the check for whether a response has anything to speak."""

import pytest

from pipecat.utils.text.alnum_utils import has_alnum


@pytest.mark.parametrize(
    "text",
    [
        "好",
        "您好，請問需要什麼協助？",
        "OK",
        # Full-width digits and letters are alphanumeric like their ASCII forms.
        "１２３",
        "ＡＢＣ",
        # A lone '<' is content, not the start of markup.
        "5 < 10",
        # Markup is stripped, the text it wraps is not.
        "<speak>您好</speak>",
        '<prosody rate="slow">好的</prosody>',
        # Text around a function tag is still spoken once the tag is removed.
        "請稍候<function=transfer_call>{}</function>",
        "<function=end_call/>再見",
        "​好",
        "😀 好的",
    ],
)
def test_speakable_text(text):
    assert has_alnum(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        " \n\t",
        "...",
        "。",
        # Full-width punctuation.
        "，！？「」",
        # SSML that wraps nothing speakable.
        "<break/>",
        '<break time="1s"/>',
        "<speak> </speak>",
        # Function-tag markup is removed whole, arguments included, before TTS.
        '<function=transfer_call>{"reason":"x"}</function>',
        "<function=end_call></function>",
        "<function=end_call/>",
        '<FUNCTION=end_call>\n{"a": 1}\n</FUNCTION>',
        "😀",
        "👍🏻",
        # Zero-width space and joiner.
        "​",
        "‍",
    ],
)
def test_unspeakable_text(text):
    assert not has_alnum(text)
