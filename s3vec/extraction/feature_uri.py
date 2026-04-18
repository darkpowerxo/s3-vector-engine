"""Feature URI system — every extracted feature is addressable.

Format: ``s3vec://extractor@version/feature_type``

Examples::

    s3vec://arcface@v1/embedding       → ArcFace 512d vector
    s3vec://siglip@v1/embedding        → SigLIP 1152d vector
    s3vec://clap@v1/fingerprint        → CLAP audio embedding
    s3vec://whisper@v1/transcript      → Whisper ASR output
    s3vec://e5_large@v1/embedding      → E5-Large 1024d vector
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_URI_PATTERN = re.compile(
    r"^s3vec://(?P<extractor>[a-zA-Z0-9_]+)@(?P<version>v\d+)/(?P<feature_type>[a-zA-Z0-9_]+)$"
)


@dataclass(frozen=True)
class FeatureURI:
    """Addressable reference to an extracted feature."""

    extractor: str
    version: str
    feature_type: str

    def __str__(self) -> str:
        return f"s3vec://{self.extractor}@{self.version}/{self.feature_type}"

    @classmethod
    def parse(cls, uri: str) -> FeatureURI:
        """Parse a feature URI string.

        Raises ``ValueError`` on invalid format.
        """
        m = _URI_PATTERN.match(uri)
        if not m:
            raise ValueError(
                f"Invalid feature URI: {uri!r}. "
                "Expected format: s3vec://extractor@version/feature_type"
            )
        return cls(**m.groupdict())
