"""TinyGPT classifier vendored from the llm-tutorial repo.

Source: /home/asudjianto/jupyterlab/llm-tutorial/lm_from_scratch/ at the
commit current as of agent-tutorial's switch from a keyword classifier to
the fine-tuned TinyGPT classifier. Files are byte-identical to upstream
except for import rewrites (lm_from_scratch.X -> glassloop.models.tinygpt.X)
so this package can run without the llm-tutorial source on sys.path.

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
