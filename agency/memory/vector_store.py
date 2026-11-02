"""Hafif vektor hafiza (harici embedding servisi gerektirmez).

Piyasa snapshot'lari sayisal ozellik sozlugu olarak saklanir. Benzer piyasa
kosullarini bulmak icin "signed hashing trick" ile TF-IDF benzeri seyrek
vektor uretilir:

    * Her ozellik, sabit bir olcege bolunerek kovaya (bucket) ayrilir.
    * ``("<ozellik>", "<kova>")`` token'i SHA-256 ile 256 boyutlu uzaya
      isaretli (signed) olarak izdusurulur.
    * Vektor L2 normalize edilir; benzerlik kosinus ile hesaplanir.

Avantaji: deterministik, cevrimdisi calisir, ek bagimlilik yok ve ayni piyasa
rejimleri birbirine yakin vektorler uretir.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

#: Ozellik -> kova buyuklugu (domain bilgisi). Listede olmayan ozellikler icin
#: log-tabanli otomatik kova kullanilir.
FEATURE_SCALES: Dict[str, float] = {
    "rsi": 5.0,
    "rsi_delta": 2.0,
    "adx": 5.0,
    "plus_di": 5.0,
    "minus_di": 5.0,
    "atr_pct": 0.5,
    "volatility_ratio": 0.15,
    "rel_volume": 0.25,
    "volume_z": 0.5,
    "macd_hist_pct": 0.05,
    "roc": 1.0,
    "bb_width": 1.0,
    "bb_pctb": 0.15,
    "stoch_k": 10.0,
    "ema_spread_pct": 0.3,
    "dist_ema_spread_pct": 0.5,
    "dist_ema_slow_pct": 1.0,
    "close_pos_20": 0.1,
    "close_pos_50": 0.1,
    "trend_score": 0.25,
    "slope_20_pct": 0.2,
    "drawdown_from_high_pct": 2.0,
    "runup_from_low_pct": 2.0,
}

CATEGORICAL_FIELDS: Sequence[str] = ("regime", "volatility_regime", "trend_strength")


def _bucket(scale: float, value: float) -> int:
    return int(round(value / scale))


def _auto_bucket(value: float) -> int:
    if value == 0:
        return 0
    magnitude = math.log1p(abs(value)) * 4.0
    return int(round(math.copysign(magnitude, value)))


class FeatureVectorizer:
    """Sayisal ozellikleri L2-normalize edilmis vektorlere cevirir."""

    def __init__(self, dim: int = 256):
        if dim < 8:
            raise ValueError("vector_dim en az 8 olmali")
        self.dim = int(dim)

    # -- token uretimi ------------------------------------------------------
    def tokens(self, features: Mapping[str, Any]) -> List[str]:
        tokens: List[str] = []
        for name, raw in features.items():
            if raw is None or isinstance(raw, bool):
                continue
            if isinstance(raw, str):
                if name in CATEGORICAL_FIELDS:
                    tokens.append(f"{name}={raw}")
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            scale = FEATURE_SCALES.get(name)
            bucket = _bucket(scale, value) if scale else _auto_bucket(value)
            tokens.append(f"{name}#{bucket}")
        return tokens

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    # -- gomme --------------------------------------------------------------
    def embed(self, features: Mapping[str, Any], *, extra_tokens: Iterable[str] = ()) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float64)
        counts: Dict[str, int] = {}
        token_list = list(self.tokens(features)) + [str(t) for t in extra_tokens]
        for token in token_list:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            index, sign = self._hash(token)
            # TF-IDF benzeri agirlik: 1 + log(tf)
            vector[index] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_dict(self, features: Mapping[str, Any]) -> List[float]:
        return [round(float(x), 6) for x in self.embed(features)]

    # -- benzerlik ----------------------------------------------------------
    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        if va.shape != vb.shape:
            return 0.0
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))

    def similarity_map(
        self,
        query_vector: Sequence[float],
        candidates: Sequence[tuple[str, Sequence[float]]],
    ) -> Dict[str, float]:
        return {key: self.cosine(query_vector, vector) for key, vector in candidates}
