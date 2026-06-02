"""Shared bounded IBLT peeling helpers."""

from __future__ import annotations

from collections.abc import Callable


def peel_iblt_cells[CellT](
    cells: tuple[CellT, ...],
    *,
    is_pure: Callable[[CellT], bool],
    is_empty: Callable[[CellT], bool],
    cell_key: Callable[[CellT], bytes],
    peel_delta: Callable[[CellT], int],
    apply_key: Callable[[list[CellT], bytes, int], None],
) -> tuple[tuple[bytes, int], ...] | None:
    """Peel IBLT cells, returning ``None`` on malformed or non-progressing sketches."""

    working = list(cells)
    peeled: list[tuple[bytes, int]] = []
    max_iterations = len(working)
    iterations = 0

    while True:
        pure_index = next(
            (index for index, cell in enumerate(working) if is_pure(cell)),
            None,
        )
        if pure_index is None:
            break
        if iterations >= max_iterations:
            return None
        iterations += 1

        cell = working[pure_index]
        key = cell_key(cell)
        delta = peel_delta(cell)
        before = working[pure_index]
        apply_key(working, key, delta)
        if working[pure_index] == before or not is_empty(working[pure_index]):
            return None
        peeled.append((key, delta))

    if any(not is_empty(cell) for cell in working):
        return None
    return tuple(peeled)
