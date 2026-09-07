"""Transient packed spatial lookup over native descriptor ordinals.

The index is an accelerator only.  Every returned ordinal is replayed against
the binary64 bounds in the immutable native descriptor pack before it can
become evidence.  Construction publishes one atomic file; interrupted partial
files are never accepted.
"""

from __future__ import annotations

from array import array
import math
import mmap
import os
from pathlib import Path
import struct


MAGIC = b"MEPSPX01"
VERSION = 3
HEADER = struct.Struct("<8sIdIQQQQQ")
CELL = struct.Struct("<iiQQ")
ORDINAL = struct.Struct("<I")
QUERY_DESCRIPTOR = struct.Struct("<5d")
INLINE_QUERY_DESCRIPTOR_FLAG = 1 << 31
DEFAULT_OVERFLOW_CELL_LIMIT = 64


def _cells(box, size):
    for x in range(math.floor(box[0] / size), math.floor(box[2] / size) + 1):
        for y in range(math.floor(box[1] / size), math.floor(box[3] / size) + 1):
            yield x, y


def _intersects(left, right):
    return (left[0] <= right[2] and right[0] <= left[2]
            and left[1] <= right[3] and right[1] <= left[3])


class PackedSpatialIndex:
    """Memory-mapped cell ranges containing descriptor-pack ordinals."""

    @classmethod
    def build(cls, path, *, descriptor_count, cell_size, descriptors,
              maximum_anchor_span_cells=1, inline_query_descriptors=True,
              descriptor_lookup=None):
        """Build atomically from ``(ordinal, exact_box, length)`` records."""
        path = Path(path)
        partial = path.with_name(path.name + ".partial")
        if path.exists() or partial.exists():
            raise ValueError("packed spatial index paths must be new")
        cell_ordinals = {}
        overflow = array("I")
        survivor_bits = bytearray((descriptor_count + 7) // 8)
        survivor_count = 0
        try:
            if inline_query_descriptors and descriptor_lookup is None:
                raise ValueError("inline query descriptors require an exact lookup")
            for ordinal, box, length in descriptors:
                if not 0 <= ordinal < descriptor_count:
                    raise ValueError("spatial ordinal outside descriptor pack")
                byte, bit = divmod(ordinal, 8)
                if survivor_bits[byte] & (1 << bit):
                    raise ValueError("duplicate spatial survivor ordinal")
                survivor_bits[byte] |= 1 << bit
                survivor_count += 1
                x0, x1 = math.floor(box[0] / cell_size), math.floor(box[2] / cell_size)
                y0, y1 = math.floor(box[1] / cell_size), math.floor(box[3] / cell_size)
                if max(x1 - x0, y1 - y0) > maximum_anchor_span_cells:
                    overflow.append(ordinal)
                    continue
                # Store every bounded primitive once, at its minimum cell.
                # Queries look back by the admitted span.  This avoids both
                # construction-time duplication and query-time deduplication.
                cell_ordinals.setdefault((x0, y0), array("I")).append(ordinal)

            keys = sorted(cell_ordinals)
            entry_count = sum(len(cell_ordinals[key]) for key in keys)
            bitmap_offset = HEADER.size + len(keys) * CELL.size
            ordinal_offset = bitmap_offset + len(survivor_bits)
            query_descriptor_offset = ordinal_offset + entry_count * ORDINAL.size
            overflow_offset = (query_descriptor_offset
                               + (entry_count * QUERY_DESCRIPTOR.size
                                  if inline_query_descriptors else 0))
            with partial.open("xb") as stream:
                encoded_span = maximum_anchor_span_cells
                if inline_query_descriptors:
                    encoded_span |= INLINE_QUERY_DESCRIPTOR_FLAG
                stream.write(HEADER.pack(
                    MAGIC, VERSION, float(cell_size), encoded_span,
                    descriptor_count,
                    survivor_count, len(keys), entry_count, len(overflow)))
                start = 0
                for x, y in keys:
                    values = cell_ordinals[(x, y)]
                    stream.write(CELL.pack(x, y, start, len(values)))
                    start += len(values)
                stream.write(survivor_bits)
                for key in keys:
                    values = cell_ordinals[key]
                    if values.itemsize != ORDINAL.size:
                        raise ValueError("platform unsigned-int width is not 32 bits")
                    stream.write(values.tobytes())
                if inline_query_descriptors:
                    query_values = array("d")
                    for key in keys:
                        for ordinal in cell_ordinals[key]:
                            box, length = descriptor_lookup(ordinal)
                            query_values.extend((*box, length))
                            if len(query_values) >= 5 * 4096:
                                stream.write(query_values.tobytes())
                                del query_values[:]
                    if query_values:
                        stream.write(query_values.tobytes())
                stream.write(overflow.tobytes())
                if inline_query_descriptors:
                    query_values = array("d")
                    for ordinal in overflow:
                        box, length = descriptor_lookup(ordinal)
                        query_values.extend((*box, length))
                        if len(query_values) >= 5 * 4096:
                            stream.write(query_values.tobytes())
                            del query_values[:]
                    if query_values:
                        stream.write(query_values.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, path)
            return cls(path)
        except BaseException:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
            raise

    def __init__(self, path):
        self.path = Path(path)
        self._stream = self.path.open("rb")
        try:
            self._map = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)
            (magic, version, self.cell_size, encoded_span,
             self.descriptor_count,
             self.survivor_count, self.cell_count, self.entry_count,
             self.overflow_count) = HEADER.unpack_from(self._map)
            if magic != MAGIC or version != VERSION or self.cell_size <= 0:
                raise ValueError("packed spatial index header mismatch")
            self.has_inline_query_descriptors = bool(
                encoded_span & INLINE_QUERY_DESCRIPTOR_FLAG)
            self.maximum_anchor_span_cells = (
                encoded_span & ~INLINE_QUERY_DESCRIPTOR_FLAG)
            self._cell_offset = HEADER.size
            self._bitmap_offset = self._cell_offset + self.cell_count * CELL.size
            self._bitmap_bytes = (self.descriptor_count + 7) // 8
            self._ordinal_offset = self._bitmap_offset + self._bitmap_bytes
            self._query_descriptor_offset = (
                self._ordinal_offset + self.entry_count * ORDINAL.size)
            self._overflow_offset = (
                self._query_descriptor_offset
                + (self.entry_count * QUERY_DESCRIPTOR.size
                   if self.has_inline_query_descriptors else 0))
            self._overflow_descriptor_offset = (
                self._overflow_offset + self.overflow_count * ORDINAL.size)
            expected = (self._overflow_descriptor_offset
                        + (self.overflow_count * QUERY_DESCRIPTOR.size
                           if self.has_inline_query_descriptors else 0))
            if len(self._map) != expected:
                raise ValueError("truncated or trailing packed spatial index data")
        except BaseException:
            self.close()
            raise

    def close(self):
        mapping = getattr(self, "_map", None)
        if mapping is not None:
            mapping.close()
            self._map = None
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def contains(self, ordinal):
        if not 0 <= ordinal < self.descriptor_count:
            return False
        byte, bit = divmod(ordinal, 8)
        return bool(self._map[self._bitmap_offset + byte] & (1 << bit))

    def survivors(self):
        for ordinal in range(self.descriptor_count):
            if self.contains(ordinal):
                yield ordinal

    def _cell_record(self, index):
        return CELL.unpack_from(self._map, self._cell_offset + index * CELL.size)

    def _cell_range(self, target):
        low, high = 0, self.cell_count
        while low < high:
            middle = (low + high) // 2
            x, y, start, count = self._cell_record(middle)
            if (x, y) < target:
                low = middle + 1
            else:
                high = middle
        if low < self.cell_count:
            x, y, start, count = self._cell_record(low)
            if (x, y) == target:
                return start, count
        return None

    def _ordinal_at(self, index):
        return ORDINAL.unpack_from(
            self._map, self._ordinal_offset + index * ORDINAL.size)[0]

    def _overflow_ordinals(self):
        for index in range(self.overflow_count):
            yield ORDINAL.unpack_from(
                self._map, self._overflow_offset + index * ORDINAL.size)[0]

    def ordinal_ranges(self, boxes):
        """Return mmap byte ranges for vectorized batch filtering."""
        ranges = []
        cells = set()
        for box in boxes:
            x0 = math.floor(box[0] / self.cell_size) - self.maximum_anchor_span_cells
            x1 = math.floor(box[2] / self.cell_size)
            y0 = math.floor(box[1] / self.cell_size) - self.maximum_anchor_span_cells
            y1 = math.floor(box[3] / self.cell_size)
            cells.update((x, y) for x in range(x0, x1 + 1)
                         for y in range(y0, y1 + 1))
        for cell in sorted(cells):
            found = self._cell_range(cell)
            if found is not None:
                start, count = found
                ranges.append((self._ordinal_offset + start * ORDINAL.size, count))
        if self.overflow_count:
            ranges.append((self._overflow_offset, self.overflow_count))
        return ranges

    def query_ranges(self, boxes):
        """Return paired ordinal and contiguous numeric-descriptor ranges."""
        if not self.has_inline_query_descriptors:
            raise ValueError("packed index has no inline query descriptors")
        output = []
        for ordinal_offset, count in self.ordinal_ranges(boxes):
            if ordinal_offset >= self._overflow_offset:
                descriptor_offset = self._overflow_descriptor_offset
            else:
                start = (ordinal_offset - self._ordinal_offset) // ORDINAL.size
                descriptor_offset = (
                    self._query_descriptor_offset + start * QUERY_DESCRIPTOR.size)
            output.append((ordinal_offset, descriptor_offset, count))
        return output

    def candidate_ordinals(self, boxes):
        """Return deduplicated coarse candidates for a batch of boxes."""
        result = set()
        for offset, count in self.ordinal_ranges(boxes):
            start = (offset - self._ordinal_offset) // ORDINAL.size
            if offset >= self._overflow_offset:
                result.update(ORDINAL.unpack_from(
                    self._map, offset + index * ORDINAL.size)[0]
                    for index in range(count))
            else:
                result.update(self._ordinal_at(index)
                              for index in range(start, start + count))
        return result

    def query_many(self, boxes, descriptor_bounds, predicate=None):
        """Batch lookup with authoritative exact-bound validation."""
        boxes = tuple(boxes)
        output = set()
        for ordinal in self.candidate_ordinals(boxes):
            exact_box, length, kind_code, style_id = descriptor_bounds(ordinal)
            if (any(_intersects(exact_box, box) for box in boxes)
                    and (predicate is None
                         or predicate(ordinal, exact_box, length, kind_code, style_id))):
                output.add(ordinal)
        return output

    def query(self, box, descriptor_bounds, predicate=None):
        return self.query_many((box,), descriptor_bounds, predicate)
