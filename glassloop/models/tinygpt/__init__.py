"""TinyGPT classifier vendored from the `lm-from-scratch` library.

Files are byte-identical to upstream (the `lm_from_scratch` package, the
companion to *Building Language Models from Scratch*) except for import
rewrites (lm_from_scratch.X -> glassloop.models.tinygpt.X) so this package can
run without the lm-from-scratch source on sys.path.

The intended public surface is narrow — only what the agent's
classify_complaint tool needs:

    from glassloop.models.tinygpt import TinyGPTClassifier, BPETokenizer
"""

from glassloop.models.tinygpt.bpe_tokenizer import BPETokenizer
from glassloop.models.tinygpt.classifier import (
    ClassifierConfig,
    TinyGPTClassifier,
)
from glassloop.models.tinygpt.configs import GPTConfig
from glassloop.models.tinygpt.gpt import TinyGPT

__all__ = [
    "BPETokenizer",
    "ClassifierConfig",
    "GPTConfig",
    "TinyGPT",
    "TinyGPTClassifier",
]
