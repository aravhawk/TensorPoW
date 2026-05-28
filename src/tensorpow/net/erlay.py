"""Deterministic Erlay transaction reconciliation sketches."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

from tensorpow.crypto.hash import HASH_LEN_BYTES, hash_bytes
from tensorpow.mempool.shard_tree import ShardId, require_shard_id

U16_BYTES: Final[int] = 2
U32_BYTES: Final[int] = 4
U64_BYTES: Final[int] = 8
U16_MAX: Final[int] = 0xFFFF
U64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF

CODEC_ERLAY: Final[int] = 0x0003
ERLAY_INTERVAL_MS: Final[int] = 8000
ERLAY_FIELD_BITS: Final[int] = 64
ERLAY_FIELD_BYTES: Final[int] = ERLAY_FIELD_BITS // 8
ERLAY_DEFAULT_CAPACITY: Final[int] = 32
ERLAY_MAX_CAPACITY: Final[int] = 1024
ERLAY_MAX_PEER_ID_BYTES: Final[int] = 512
ERLAY_SKETCH_MAGIC: Final[bytes] = b"TPERLAY"
ERLAY_SKETCH_HEADER_BYTES: Final[int] = (
    len(ERLAY_SKETCH_MAGIC) + U16_BYTES + U16_BYTES + U32_BYTES + U16_BYTES + U16_BYTES
)
ERLAY_TOPIC_TXS_PREFIX: Final[str] = "tensorpow/txs/"
ERLAY_TOPIC_TXS_SUFFIX: Final[str] = "/main"

_FIELD_POLY_LOW: Final[int] = 0x1B
_FIELD_MASK: Final[int] = U64_MAX
_SPLIT_SEED_STEP: Final[int] = 0x9E3779B97F4A7C15
_MAX_SPLIT_ATTEMPTS: Final[int] = 512
_SESSION_PREFIX: Final[bytes] = b"TensorPoW:Erlay:peer-session:"
_SHORT_ID_PREFIX: Final[bytes] = b"TensorPoW:Erlay:short-id:"


class ErlaySketchError(ValueError):
    """Raised when an Erlay sketch or reconciliation item is malformed."""


@dataclass(frozen=True, slots=True)
class ErlaySetDifference:
    """Decoded symmetric difference partitioned by the receiver's local mempool."""

    local_only_tx_ids: tuple[bytes, ...]
    remote_only_short_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_tx_id_tuple("local_only_tx_ids", self.local_only_tx_ids)
        _require_short_id_tuple("remote_only_short_ids", self.remote_only_short_ids)


@dataclass(frozen=True, slots=True)
class ErlaySketch:
    """Bounded Minisketch-style power-sum sketch over deterministic tx short IDs."""

    shard_id: ShardId
    capacity: int
    syndromes: tuple[int, ...]

    def __post_init__(self) -> None:
        require_shard_id(self.shard_id)
        _require_capacity(self.capacity)
        if not isinstance(self.syndromes, tuple):
            raise TypeError("syndromes must be a tuple")
        if len(self.syndromes) != _syndrome_count(self.capacity):
            raise ErlaySketchError("Erlay sketch syndrome count is invalid")
        for syndrome in self.syndromes:
            _require_field_element("syndrome", syndrome, allow_zero=True)

    @classmethod
    def from_short_ids(
        cls,
        shard_id: ShardId,
        short_ids: Iterable[int],
        *,
        capacity: int = ERLAY_DEFAULT_CAPACITY,
    ) -> ErlaySketch:
        """Build a canonical sketch from unique nonzero field elements."""

        shard_id = require_shard_id(shard_id)
        capacity = _require_capacity(capacity)
        canonical_short_ids = _canonical_short_ids(short_ids)
        return cls(
            shard_id=shard_id,
            capacity=capacity,
            syndromes=_syndromes_for_short_ids(canonical_short_ids, capacity),
        )

    @classmethod
    def from_tx_ids(
        cls,
        shard_id: ShardId,
        tx_ids: Iterable[bytes],
        *,
        peer_state: ErlayPeerState,
        capacity: int = ERLAY_DEFAULT_CAPACITY,
    ) -> ErlaySketch:
        """Build a canonical sketch from transaction IDs for one peer session."""

        if not isinstance(peer_state, ErlayPeerState):
            raise TypeError("peer_state must be ErlayPeerState")
        tx_id_tuple = _canonical_tx_ids(tx_ids)
        short_ids = tuple(peer_state.short_id(shard_id, tx_id) for tx_id in tx_id_tuple)
        return cls.from_short_ids(shard_id, short_ids, capacity=capacity)

    @classmethod
    def from_bytes(cls, data: bytes) -> ErlaySketch:
        """Decode canonical Erlay sketch bytes and reject malformed fields."""

        _require_bytes("data", data)
        if len(data) < ERLAY_SKETCH_HEADER_BYTES:
            raise ErlaySketchError("Erlay sketch is truncated")
        if not data.startswith(ERLAY_SKETCH_MAGIC):
            raise ErlaySketchError("Erlay sketch magic is invalid")

        offset = len(ERLAY_SKETCH_MAGIC)
        codec_id = _read_u16(data, offset)
        offset += U16_BYTES
        if codec_id != CODEC_ERLAY:
            raise ErlaySketchError("Erlay sketch codec id is invalid")

        field_bits = _read_u16(data, offset)
        offset += U16_BYTES
        if field_bits != ERLAY_FIELD_BITS:
            raise ErlaySketchError("Erlay sketch field encoding is invalid")

        shard_id = _read_u32(data, offset)
        offset += U32_BYTES
        try:
            require_shard_id(shard_id)
        except (TypeError, ValueError) as exc:
            raise ErlaySketchError("Erlay sketch shard id is invalid") from exc

        capacity = _read_u16(data, offset)
        offset += U16_BYTES
        try:
            _require_capacity(capacity)
        except ValueError as exc:
            raise ErlaySketchError("Erlay sketch capacity is invalid") from exc

        syndrome_count = _read_u16(data, offset)
        offset += U16_BYTES
        if syndrome_count != _syndrome_count(capacity):
            raise ErlaySketchError("Erlay sketch syndrome count is invalid")

        expected_len = ERLAY_SKETCH_HEADER_BYTES + syndrome_count * ERLAY_FIELD_BYTES
        if len(data) != expected_len:
            raise ErlaySketchError("Erlay sketch length is invalid")

        syndromes = tuple(
            _read_u64(data, offset + index * ERLAY_FIELD_BYTES) for index in range(syndrome_count)
        )
        return cls(shard_id=shard_id, capacity=capacity, syndromes=syndromes)

    def to_bytes(self) -> bytes:
        """Return the canonical wire bytes for this Erlay sketch."""

        parts = [
            ERLAY_SKETCH_MAGIC,
            _u16(CODEC_ERLAY),
            _u16(ERLAY_FIELD_BITS),
            _u32(self.shard_id),
            _u16(self.capacity),
            _u16(len(self.syndromes)),
        ]
        parts.extend(_u64(syndrome) for syndrome in self.syndromes)
        return b"".join(parts)

    def difference(self, other: ErlaySketch) -> ErlaySketch:
        """Return the XOR sketch for the symmetric difference with ``other``."""

        if not isinstance(other, ErlaySketch):
            raise TypeError("other must be ErlaySketch")
        if self.shard_id != other.shard_id:
            raise ErlaySketchError("Erlay sketches must use the same shard")
        if self.capacity != other.capacity:
            raise ErlaySketchError("Erlay sketches must use the same capacity")
        return ErlaySketch(
            shard_id=self.shard_id,
            capacity=self.capacity,
            syndromes=tuple(
                left ^ right for left, right in zip(self.syndromes, other.syndromes, strict=True)
            ),
        )

    def decode_short_ids(self) -> tuple[int, ...]:
        """Decode this sketch as a bounded symmetric-difference set."""

        short_ids = _decode_syndromes(self.syndromes, self.capacity)
        if _syndromes_for_short_ids(short_ids, self.capacity) != self.syndromes:
            raise ErlaySketchError("Erlay sketch does not decode canonically")
        return short_ids


@dataclass(slots=True)
class ErlayPeerState:
    """Per-peer deterministic Erlay session state with per-shard round timing."""

    local_peer_id: bytes
    remote_peer_id: bytes
    interval_ms: int = ERLAY_INTERVAL_MS
    _last_reconciled_ms_by_shard: dict[ShardId, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_peer_id",
            _require_peer_id("local_peer_id", self.local_peer_id),
        )
        object.__setattr__(
            self,
            "remote_peer_id",
            _require_peer_id("remote_peer_id", self.remote_peer_id),
        )
        if self.local_peer_id == self.remote_peer_id:
            raise ValueError("peer ids must be distinct")
        object.__setattr__(
            self,
            "interval_ms",
            _require_positive_int("interval_ms", self.interval_ms),
        )

    @property
    def session_id(self) -> bytes:
        """Return the order-independent deterministic session identifier."""

        first, second = sorted((self.local_peer_id, self.remote_peer_id))
        return hash_bytes(
            b"".join(
                (
                    _SESSION_PREFIX,
                    _u16(len(first)),
                    first,
                    _u16(len(second)),
                    second,
                )
            )
        )

    def topic_for_shard(self, shard_id: ShardId) -> str:
        """Return the canonical gossipsub transaction topic for one shard."""

        return erlay_topic_for_shard(shard_id)

    def short_id(self, shard_id: ShardId, tx_id: bytes) -> int:
        """Return the nonzero deterministic 64-bit reconciliation ID for ``tx_id``."""

        shard_id = require_shard_id(shard_id)
        _require_tx_id("tx_id", tx_id)
        counter = 0
        while counter <= U16_MAX:
            digest = hash_bytes(
                b"".join(
                    (
                        _SHORT_ID_PREFIX,
                        self.session_id,
                        _u32(shard_id),
                        tx_id,
                        _u16(counter),
                    )
                )
            )
            short_id = int.from_bytes(digest[:ERLAY_FIELD_BYTES], "little")
            if short_id != 0:
                return short_id
            counter += 1
        raise ErlaySketchError("unable to derive nonzero Erlay short id")

    def build_sketch(
        self,
        shard_id: ShardId,
        tx_ids: Iterable[bytes],
        *,
        capacity: int = ERLAY_DEFAULT_CAPACITY,
    ) -> ErlaySketch:
        """Build this peer session's canonical sketch for one shard."""

        return ErlaySketch.from_tx_ids(shard_id, tx_ids, peer_state=self, capacity=capacity)

    def decode_difference(self, local: ErlaySketch, remote: ErlaySketch | bytes) -> tuple[int, ...]:
        """Decode the short IDs in ``local`` XOR ``remote``."""

        remote_sketch = ErlaySketch.from_bytes(remote) if isinstance(remote, bytes) else remote
        if not isinstance(local, ErlaySketch):
            raise TypeError("local must be ErlaySketch")
        return local.difference(remote_sketch).decode_short_ids()

    def reconcile(
        self,
        shard_id: ShardId,
        local_tx_ids: Iterable[bytes],
        local_sketch: ErlaySketch,
        remote_sketch: ErlaySketch | bytes,
    ) -> ErlaySetDifference:
        """Decode and partition a peer sketch against the local tx-id set."""

        shard_id = require_shard_id(shard_id)
        if local_sketch.shard_id != shard_id:
            raise ErlaySketchError("local sketch shard does not match requested shard")
        difference_short_ids = self.decode_difference(local_sketch, remote_sketch)
        return self.partition_difference(shard_id, local_tx_ids, difference_short_ids)

    def partition_difference(
        self,
        shard_id: ShardId,
        local_tx_ids: Iterable[bytes],
        difference_short_ids: Iterable[int],
    ) -> ErlaySetDifference:
        """Split decoded short IDs into local-only tx IDs and remote-only short IDs."""

        short_ids = _canonical_short_ids(difference_short_ids)
        local_by_short_id = self.index_tx_ids_by_short_id(shard_id, local_tx_ids)
        local_only: list[bytes] = []
        remote_only: list[int] = []
        for short_id in short_ids:
            tx_id = local_by_short_id.get(short_id)
            if tx_id is None:
                remote_only.append(short_id)
            else:
                local_only.append(tx_id)
        return ErlaySetDifference(
            local_only_tx_ids=tuple(sorted(local_only)),
            remote_only_short_ids=tuple(remote_only),
        )

    def resolve_short_ids(
        self,
        shard_id: ShardId,
        local_tx_ids: Iterable[bytes],
        requested_short_ids: Iterable[int],
    ) -> tuple[bytes, ...]:
        """Resolve requested short IDs to local tx IDs for serving a peer."""

        short_ids = _canonical_short_ids(requested_short_ids)
        local_by_short_id = self.index_tx_ids_by_short_id(shard_id, local_tx_ids)
        resolved: list[bytes] = []
        for short_id in short_ids:
            tx_id = local_by_short_id.get(short_id)
            if tx_id is None:
                raise ErlaySketchError("requested Erlay short id is not in the local set")
            resolved.append(tx_id)
        return tuple(resolved)

    def index_tx_ids_by_short_id(
        self,
        shard_id: ShardId,
        tx_ids: Iterable[bytes],
    ) -> dict[int, bytes]:
        """Return a collision-checked map from short ID to tx ID for one shard."""

        shard_id = require_shard_id(shard_id)
        tx_id_tuple = _canonical_tx_ids(tx_ids)
        index: dict[int, bytes] = {}
        for tx_id in tx_id_tuple:
            short_id = self.short_id(shard_id, tx_id)
            existing = index.get(short_id)
            if existing is not None and existing != tx_id:
                raise ErlaySketchError("Erlay short id collision in local tx set")
            index[short_id] = tx_id
        return index

    def should_reconcile(self, shard_id: ShardId, now_ms: int) -> bool:
        """Return whether this peer is due for a reconciliation round on ``shard_id``."""

        shard_id = require_shard_id(shard_id)
        now_ms = _require_nonnegative_int("now_ms", now_ms)
        last_ms = self._last_reconciled_ms_by_shard.get(shard_id)
        return last_ms is None or now_ms - last_ms >= self.interval_ms

    def mark_reconciled(self, shard_id: ShardId, now_ms: int) -> None:
        """Record a completed reconciliation round for one shard."""

        shard_id = require_shard_id(shard_id)
        now_ms = _require_nonnegative_int("now_ms", now_ms)
        previous = self._last_reconciled_ms_by_shard.get(shard_id)
        if previous is not None and now_ms < previous:
            raise ValueError("now_ms must not move backwards for a shard")
        self._last_reconciled_ms_by_shard[shard_id] = now_ms

    def next_reconcile_at_ms(self, shard_id: ShardId) -> int:
        """Return the next due timestamp for one shard, or zero if never run."""

        shard_id = require_shard_id(shard_id)
        last_ms = self._last_reconciled_ms_by_shard.get(shard_id)
        if last_ms is None:
            return 0
        return last_ms + self.interval_ms


def erlay_topic_for_shard(shard_id: ShardId) -> str:
    """Return `TOPIC_TXS_PREFIX || shard_id_hex || TOPIC_TXS_SUFFIX`."""

    shard_id = require_shard_id(shard_id)
    return f"{ERLAY_TOPIC_TXS_PREFIX}{shard_id:08x}{ERLAY_TOPIC_TXS_SUFFIX}"


def _decode_syndromes(syndromes: tuple[int, ...], capacity: int) -> tuple[int, ...]:
    if all(syndrome == 0 for syndrome in syndromes):
        return ()

    locator = _berlekamp_massey(syndromes)
    degree = len(locator) - 1
    if degree == 0 or degree > capacity:
        raise ErlaySketchError("Erlay sketch exceeds reconciliation capacity")

    item_polynomial = _item_polynomial_from_locator(locator)
    if _poly_gcd(item_polynomial, _poly_derivative(item_polynomial)) != (1,):
        raise ErlaySketchError("Erlay sketch decodes to duplicate short ids")

    roots = tuple(sorted(_split_linear_roots(item_polynomial)))
    if len(roots) != degree:
        raise ErlaySketchError("Erlay sketch does not split into field elements")
    for root in roots:
        _require_field_element("decoded short id", root, allow_zero=False)
    return roots


def _berlekamp_massey(sequence: tuple[int, ...]) -> tuple[int, ...]:
    connection = [0] * (len(sequence) + 1)
    previous = [0] * (len(sequence) + 1)
    connection[0] = 1
    previous[0] = 1
    degree = 0
    gap = 1
    last_discrepancy = 1

    for index, syndrome in enumerate(sequence):
        discrepancy = syndrome
        for coefficient_index in range(1, degree + 1):
            discrepancy ^= _field_mul(
                connection[coefficient_index],
                sequence[index - coefficient_index],
            )
        if discrepancy == 0:
            gap += 1
            continue

        before_update = connection.copy()
        scale = _field_div(discrepancy, last_discrepancy)
        for coefficient_index, coefficient in enumerate(previous[: len(sequence) + 1 - gap]):
            connection[coefficient_index + gap] ^= _field_mul(scale, coefficient)

        if 2 * degree <= index:
            degree = index + 1 - degree
            previous = before_update
            last_discrepancy = discrepancy
            gap = 1
        else:
            gap += 1

    return tuple(connection[: degree + 1])


def _item_polynomial_from_locator(locator: tuple[int, ...]) -> tuple[int, ...]:
    degree = len(locator) - 1
    if degree <= 0:
        return (0,)
    return _poly_trim((*tuple(locator[degree - index] for index in range(degree)), 1))


def _split_linear_roots(polynomial: tuple[int, ...]) -> tuple[int, ...]:
    polynomial = _poly_make_monic(polynomial)
    degree = _poly_degree(polynomial)
    if degree == 0:
        return ()
    if degree == 1:
        return (_field_div(polynomial[0], polynomial[1]),)

    for attempt in range(1, _MAX_SPLIT_ATTEMPTS + 1):
        seed = (attempt * _SPLIT_SEED_STEP) & _FIELD_MASK
        if seed == 0:
            continue
        trace = _poly_trace_mod((0, seed), polynomial)
        factor = _poly_gcd(polynomial, trace)
        factor_degree = _poly_degree(factor)
        if 0 < factor_degree < degree:
            quotient, remainder = _poly_divmod(polynomial, factor)
            if remainder != (0,):
                raise ErlaySketchError("Erlay polynomial factorization failed")
            return _split_linear_roots(factor) + _split_linear_roots(quotient)
    raise ErlaySketchError("Erlay sketch polynomial did not split")


def _poly_trace_mod(base: tuple[int, ...], modulus: tuple[int, ...]) -> tuple[int, ...]:
    result: tuple[int, ...] = (0,)
    value = _poly_mod(base, modulus)
    for _ in range(ERLAY_FIELD_BITS):
        result = _poly_add(result, value)
        value = _poly_mul_mod(value, value, modulus)
    return result


def _poly_add(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    size = max(len(left), len(right))
    return _poly_trim(
        tuple(
            (left[index] if index < len(left) else 0) ^ (right[index] if index < len(right) else 0)
            for index in range(size)
        )
    )


def _poly_mul(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    if left == (0,) or right == (0,):
        return (0,)
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_coefficient in enumerate(left):
        if left_coefficient == 0:
            continue
        for right_index, right_coefficient in enumerate(right):
            if right_coefficient != 0:
                result[left_index + right_index] ^= _field_mul(
                    left_coefficient,
                    right_coefficient,
                )
    return _poly_trim(tuple(result))


def _poly_mul_mod(
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: tuple[int, ...],
) -> tuple[int, ...]:
    return _poly_mod(_poly_mul(left, right), modulus)


def _poly_mod(value: tuple[int, ...], modulus: tuple[int, ...]) -> tuple[int, ...]:
    return _poly_divmod(value, modulus)[1]


def _poly_divmod(
    dividend: tuple[int, ...],
    divisor: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dividend = _poly_trim(dividend)
    divisor = _poly_trim(divisor)
    if divisor == (0,):
        raise ZeroDivisionError("polynomial division by zero")
    if _poly_degree(dividend) < _poly_degree(divisor):
        return (0,), dividend

    quotient = [0] * (_poly_degree(dividend) - _poly_degree(divisor) + 1)
    remainder = list(dividend)
    inverse_lead = _field_inv(divisor[-1])

    while len(remainder) >= len(divisor) and tuple(remainder) != (0,):
        shift = len(remainder) - len(divisor)
        scale = _field_mul(remainder[-1], inverse_lead)
        quotient[shift] = scale
        if scale != 0:
            for index, coefficient in enumerate(divisor):
                if coefficient != 0:
                    remainder[index + shift] ^= _field_mul(scale, coefficient)
        remainder = list(_poly_trim(tuple(remainder)))

    return _poly_trim(tuple(quotient)), _poly_trim(tuple(remainder))


def _poly_gcd(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    left = _poly_trim(left)
    right = _poly_trim(right)
    while right != (0,):
        left, right = right, _poly_mod(left, right)
    return _poly_make_monic(left)


def _poly_derivative(polynomial: tuple[int, ...]) -> tuple[int, ...]:
    if len(polynomial) <= 1:
        return (0,)
    return _poly_trim(
        tuple(polynomial[index] if index % 2 == 1 else 0 for index in range(1, len(polynomial)))
    )


def _poly_make_monic(polynomial: tuple[int, ...]) -> tuple[int, ...]:
    polynomial = _poly_trim(polynomial)
    if polynomial == (0,) or polynomial[-1] == 1:
        return polynomial
    inverse_lead = _field_inv(polynomial[-1])
    return _poly_trim(tuple(_field_mul(coefficient, inverse_lead) for coefficient in polynomial))


def _poly_degree(polynomial: tuple[int, ...]) -> int:
    return len(_poly_trim(polynomial)) - 1


def _poly_trim(polynomial: tuple[int, ...]) -> tuple[int, ...]:
    values = list(polynomial)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    if not values:
        return (0,)
    return tuple(values)


def _syndromes_for_short_ids(short_ids: tuple[int, ...], capacity: int) -> tuple[int, ...]:
    count = _syndrome_count(capacity)
    syndromes = [0] * count
    for short_id in short_ids:
        power = short_id
        for index in range(count):
            syndromes[index] ^= power
            power = _field_mul(power, short_id)
    return tuple(syndromes)


def _canonical_tx_ids(tx_ids: Iterable[bytes]) -> tuple[bytes, ...]:
    values = tuple(tx_ids)
    for tx_id in values:
        _require_tx_id("tx_id", tx_id)
    ordered = tuple(sorted(values))
    if len(set(ordered)) != len(ordered):
        raise ErlaySketchError("Erlay tx_id set contains duplicates")
    return ordered


def _canonical_short_ids(short_ids: Iterable[int]) -> tuple[int, ...]:
    if isinstance(short_ids, int):
        raise TypeError("short_ids must be an iterable of int values")
    values = tuple(short_ids)
    for short_id in values:
        _require_field_element("short_id", short_id, allow_zero=False)
    ordered = tuple(sorted(values))
    if len(set(ordered)) != len(ordered):
        raise ErlaySketchError("Erlay short id set contains duplicates")
    return ordered


def _require_tx_id_tuple(name: str, values: tuple[bytes, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ErlaySketchError(f"{name} must be canonical")
    for value in values:
        _require_tx_id(name, value)


def _require_short_id_tuple(name: str, values: tuple[int, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ErlaySketchError(f"{name} must be canonical")
    for value in values:
        _require_field_element(name, value, allow_zero=False)


def _syndrome_count(capacity: int) -> int:
    capacity = _require_capacity(capacity)
    return 2 * capacity


def _field_mul(left: int, right: int) -> int:
    _require_field_element("left", left, allow_zero=True)
    _require_field_element("right", right, allow_zero=True)
    result = 0
    addend = left
    multiplier = right
    while multiplier:
        if multiplier & 1:
            result ^= addend
        multiplier >>= 1
        addend <<= 1
        if addend >> ERLAY_FIELD_BITS:
            addend = (addend ^ _FIELD_POLY_LOW) & _FIELD_MASK
    return result & _FIELD_MASK


def _field_pow(value: int, exponent: int) -> int:
    _require_field_element("value", value, allow_zero=True)
    _require_nonnegative_int("exponent", exponent)
    result = 1
    base = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _field_mul(result, base)
        remaining >>= 1
        if remaining:
            base = _field_mul(base, base)
    return result


def _field_inv(value: int) -> int:
    _require_field_element("value", value, allow_zero=False)
    return _field_pow(value, (1 << ERLAY_FIELD_BITS) - 2)


def _field_div(left: int, right: int) -> int:
    _require_field_element("left", left, allow_zero=True)
    return _field_mul(left, _field_inv(right))


def _require_capacity(value: int) -> int:
    value = _require_positive_int("capacity", value)
    if value > ERLAY_MAX_CAPACITY:
        raise ValueError("capacity exceeds ERLAY_MAX_CAPACITY")
    if 2 * value > U16_MAX:
        raise ValueError("capacity syndrome count exceeds uint16")
    return value


def _require_field_element(name: str, value: int, *, allow_zero: bool) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= U64_MAX:
        raise ValueError(f"{name} outside Erlay field range")
    if not allow_zero and value == 0:
        raise ErlaySketchError(f"{name} must be nonzero")


def _require_tx_id(name: str, value: bytes) -> None:
    _require_bytes(name, value)
    if len(value) != HASH_LEN_BYTES:
        raise ErlaySketchError(f"{name} must be {HASH_LEN_BYTES} bytes")


def _require_peer_id(name: str, value: bytes) -> bytes:
    _require_bytes(name, value)
    if len(value) == 0:
        raise ValueError(f"{name} must not be empty")
    if len(value) > ERLAY_MAX_PEER_ID_BYTES:
        raise ValueError(f"{name} exceeds maximum length")
    return value


def _require_bytes(name: str, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")


def _require_positive_int(name: str, value: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _read_u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + U16_BYTES], "little")


def _read_u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + U32_BYTES], "little")


def _read_u64(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + U64_BYTES], "little")


def _u16(value: int) -> bytes:
    if not 0 <= value <= U16_MAX:
        raise ValueError("value outside uint16 range")
    return value.to_bytes(U16_BYTES, "little")


def _u32(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("value outside uint32 range")
    return value.to_bytes(U32_BYTES, "little")


def _u64(value: int) -> bytes:
    if not 0 <= value <= U64_MAX:
        raise ValueError("value outside uint64 range")
    return value.to_bytes(U64_BYTES, "little")


__all__ = [
    "CODEC_ERLAY",
    "ERLAY_DEFAULT_CAPACITY",
    "ERLAY_FIELD_BITS",
    "ERLAY_INTERVAL_MS",
    "ERLAY_MAX_CAPACITY",
    "ERLAY_SKETCH_HEADER_BYTES",
    "ERLAY_SKETCH_MAGIC",
    "ErlayPeerState",
    "ErlaySetDifference",
    "ErlaySketch",
    "ErlaySketchError",
    "erlay_topic_for_shard",
]
