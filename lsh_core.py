"""
lsh_core.py
===========
Shared MinHash and LSH classes used by both build_index.py and pipeline.py.
Must be a standalone module so pickle can resolve the class references.
"""

import hashlib
import numpy as np

# ── Configuration defaults (overridable at construction) ──────────────────────
N_HASHES     = 128
N_BANDS      = 16
SHINGLE_SIZE = 3


class MinHasher:
    """
    MinHash signature generator.

    Uses the universal hash family:  h(x) = (a*x + b) mod p
    where p is a Mersenne prime larger than the hash space.

    Parameters
    ----------
    n_hashes : int
        Length of the MinHash signature vector.
    seed : int
        Random seed for reproducibility.
    """

    MERSENNE_PRIME = (1 << 61) - 1   # 2^61 - 1

    def __init__(self, n_hashes: int = N_HASHES, seed: int = 42):
        self.n_hashes = n_hashes
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, self.MERSENNE_PRIME, size=n_hashes, dtype=np.int64)
        self.b = rng.randint(0, self.MERSENNE_PRIME, size=n_hashes, dtype=np.int64)

    def _hash_token(self, token: str) -> int:
        """Map a string token to a non-negative integer via MD5."""
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.MERSENNE_PRIME

    def get_shingles(self, text: str, k: int = SHINGLE_SIZE) -> set:
        """Return the set of word k-shingles from `text`."""
        words = text.lower().split()
        if len(words) < k:
            return set(words)
        return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}

    def signature(self, text: str) -> np.ndarray:
        """
        Compute the MinHash signature for `text`.

        For each hash function h_i:
            signature[i] = min_{s in shingles} h_i(s)

        This approximates Jaccard similarity:
            P[sig_A[i] == sig_B[i]]  ≈  Jaccard(A, B)
        """
        shingles = self.get_shingles(text)
        p   = self.MERSENNE_PRIME
        sig = np.full(self.n_hashes, p, dtype=np.int64)
        for shingle in shingles:
            x      = self._hash_token(shingle)
            hashes = (self.a * x + self.b) % p
            sig    = np.minimum(sig, hashes)
        return sig


class LSHIndex:
    """
    LSH index using the banding technique.

    A signature of length `n_hashes` is divided into `n_bands` bands, each
    of `rows_per_band` rows.  Two documents end up in the same bucket for
    band b iff their band-b sub-signatures are identical.

    Theoretical Jaccard threshold at which collision probability = 0.5:
        s* = (1 / n_bands) ^ (1 / rows_per_band)

    With n_hashes=128, n_bands=16  =>  s* ~= 0.707

    Parameters
    ----------
    n_hashes : int
        Must match the MinHasher's n_hashes.
    n_bands : int
        Number of bands. Must divide n_hashes evenly.
    """

    def __init__(self, n_hashes: int = N_HASHES, n_bands: int = N_BANDS):
        assert n_hashes % n_bands == 0, \
            f"n_hashes ({n_hashes}) must be divisible by n_bands ({n_bands})"
        self.n_hashes      = n_hashes
        self.n_bands       = n_bands
        self.rows_per_band = n_hashes // n_bands
        # buckets[b] : dict[bucket_key -> list[chunk_idx]]
        self.buckets = [{} for _ in range(n_bands)]

    @property
    def threshold(self) -> float:
        """Theoretical Jaccard threshold s* where collision probability = 0.5."""
        return (1 / self.n_bands) ** (1 / self.rows_per_band)

    def _band_key(self, signature: np.ndarray, band: int) -> int:
        start = band * self.rows_per_band
        end   = start + self.rows_per_band
        return hash(tuple(signature[start:end].tolist()))

    def add(self, idx: int, signature: np.ndarray) -> None:
        """Insert chunk index `idx` with its MinHash `signature`."""
        for b in range(self.n_bands):
            key = self._band_key(signature, b)
            self.buckets[b].setdefault(key, []).append(idx)

    def query(self, signature: np.ndarray) -> set:
        """
        Return the set of candidate chunk indices that share at least one
        band bucket with the given query signature.
        """
        candidates: set = set()
        for b in range(self.n_bands):
            key = self._band_key(signature, b)
            if key in self.buckets[b]:
                candidates.update(self.buckets[b][key])
        return candidates
