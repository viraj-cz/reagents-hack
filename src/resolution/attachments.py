"""Uploaded tables: stored once, profiled once, then used three ways.

A file the user attaches has to satisfy three consumers that want very
different things from it, which is why this module exists rather than a `files`
list threaded through `build_problem`:

* **GOD** needs *evidence it can project*. Raw rows cannot go into
  `NativeProblem.inputs`: `transformer._source_units` explodes that dict into
  one projection obligation per leaf value, and the manifest check demands a
  mapping for every one. A 10k-row CSV would ask the model to author 10k
  mappings and fail validation forever. So GOD gets `profile()` -- shape,
  column types, ranges, category lists -- which is a couple of dozen units.
* **The seal** needs *native vocabulary to hide*. Column names and the values
  of low-cardinality columns are exactly the native labels a demigod must not
  see, and `sealed_terms()` derives them mechanically. This replaces the
  hand-typed entity box: nobody should have to retype their own schema.
* **The demigod** needs *the bytes*, mounted read-only under `shared/` so
  pandas can actually compute on them.

WHAT THE PROFILE DOES NOT DO
----------------------------
It does not make an attached run sealed. The raw file is mounted into the
sandbox with its native headers intact, exactly as `seed_shared_files` warns:
"a CSV with native column headers walks straight past the seal that the
envelope check enforces." Deriving the terms still buys the two GOD-side
defences -- the planner is anonymised and the transformer's leak check has
teeth -- but a demigod that reads the file sees the real column names. Callers
should say so rather than report such a run as fully sealed.
"""

from __future__ import annotations

import csv
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Every cap here exists to bound the projection manifest, not to save memory.
# See the module docstring: each surviving column costs one obligation the
# transformer must author and the validator must match.
MAX_COLUMNS = 40

# A column with no more distinct values than this is treated as categorical:
# its values are listed in the profile and become sealed terms. Above it, the
# column is summarised by range instead -- listing 5000 order ids would neither
# help GOD reason nor produce a usable seal.
CATEGORICAL_MAX_UNIQUE = 24

# Distinct values kept per categorical column. Inside one column dict, so this
# costs profile size but never an extra obligation.
MAX_SAMPLE_VALUES = 24

# Rows scanned for statistics. The row COUNT is always exact -- it is cheap and
# a wrong one is the kind of quiet lie that survives review. Only the stats are
# drawn from the first slice.
MAX_ROWS_SCANNED = 50_000

# Separates the category list inside a column's `values` string. Chosen over a
# comma because a value containing ", " would otherwise split into two sealed
# terms -- harmless (over-sealing hides more, it never leaks less) but confusing
# to read back in the UI.
VALUE_SEPARATOR = " | "

# Upload ceiling. Generous for a table you want summarised, small enough that a
# careless drop of a multi-gigabyte parquet fails fast with a readable message.
MAX_UPLOAD_BYTES = 256 * 1024 * 1024

SUPPORTED_SUFFIXES = (".csv", ".tsv", ".parquet")

# Column names too generic to be worth sealing. Hiding "date" or "id" costs
# real planning signal -- the anonymiser rewrites every occurrence in the prose
# too, so GOD would read "e4 of each e7" -- while revealing nothing about which
# field the problem came from. The seal is meant to hide the problem's
# identity, and these words identify nothing.
_GENERIC_COLUMNS = frozenset(
    {
        "id",
        "index",
        "idx",
        "key",
        "name",
        "date",
        "datetime",
        "timestamp",
        "time",
        "year",
        "month",
        "day",
        "week",
        "quarter",
        "hour",
        "minute",
        "value",
        "values",
        "count",
        "total",
        "sum",
        "amount",
        "type",
        "category",
        "label",
        "status",
        "code",
        "notes",
        "note",
        "comment",
        "description",
        "unit",
        "units",
    }
)

_TRUE = {"true", "yes", "y", "t", "1"}
_FALSE = {"false", "no", "n", "f", "0"}


class AttachmentError(ValueError):
    """Bad upload, reported to the browser as a 400 rather than a traceback."""


@dataclass
class Attachment:
    """One uploaded table: the bytes on disk, plus what we learned from them."""

    id: str
    name: str
    size: int
    path: Path
    profile: dict[str, Any] = field(default_factory=dict)
    terms: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "profile": self.profile,
            "terms": list(self.terms),
        }


class AttachmentStore:
    """Uploads for this server process, keyed by id.

    Files live in a directory the caller owns rather than a `TemporaryDirectory`
    that vanishes on garbage collection: a run seeds them into a Modal volume
    well after the upload request has returned, and a path that disappears
    between those two moments is a race nobody would enjoy debugging.
    """

    def __init__(self, root: Path | None = None, *, max_files: int = 64) -> None:
        self.root = root or (Path.home() / ".cache" / "resolution" / "uploads")
        self.max_files = max_files
        self.items: dict[str, Attachment] = {}
        self.order: list[str] = []

    def add(self, name: str, data: bytes) -> Attachment:
        clean = _safe_name(name)
        suffix = Path(clean).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise AttachmentError(
                f"unsupported file type '{suffix or clean}': "
                f"attach one of {', '.join(SUPPORTED_SUFFIXES)}"
            )
        if not data:
            raise AttachmentError(f"{clean} is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise AttachmentError(
                f"{clean} is {_human(len(data))}; the limit is "
                f"{_human(MAX_UPLOAD_BYTES)}"
            )

        attachment_id = f"att-{uuid.uuid4().hex[:8]}"
        # One directory per upload, so two files called `data.csv` can both be
        # attached to the same run without one overwriting the other -- and so
        # the name a demigod sees under shared/ is the name the user chose.
        folder = self.root / attachment_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / clean
        path.write_bytes(data)

        profile = profile_table(path)
        item = Attachment(
            id=attachment_id,
            name=clean,
            size=len(data),
            path=path,
            profile=profile,
            terms=sealed_terms(profile),
        )
        self.items[attachment_id] = item
        self.order.append(attachment_id)
        self._evict()
        return item

    def get(self, attachment_id: str) -> Attachment | None:
        return self.items.get(attachment_id)

    def resolve(self, ids: list[str]) -> list[Attachment]:
        found: list[Attachment] = []
        for attachment_id in ids:
            item = self.get(str(attachment_id))
            if item is None:
                raise AttachmentError(f"unknown attachment: {attachment_id}")
            found.append(item)
        return found

    def _evict(self) -> None:
        while len(self.order) > self.max_files:
            stale = self.order.pop(0)
            item = self.items.pop(stale, None)
            if item is None:
                continue
            # Best effort: a run still holding this path is rare (it would have
            # to outlive `max_files` later uploads) and losing the bytes is
            # better than growing the cache without bound.
            try:
                item.path.unlink(missing_ok=True)
                item.path.parent.rmdir()
            except OSError:
                pass


# -- profiling -------------------------------------------------------------
def profile_table(path: Path) -> dict[str, Any]:
    """Compact, projectable description of a table. Never its rows."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _profile_parquet(path)
    return _profile_delimited(path, delimiter="\t" if suffix == ".tsv" else ",")


def _profile_delimited(path: Path, *, delimiter: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise AttachmentError(f"{path.name} has no header row") from exc

        names = _dedupe(
            [(cell.strip() or f"column_{i + 1}") for i, cell in enumerate(header)]
        )
        kept = names[:MAX_COLUMNS]
        columns = [_ColumnStats(name) for name in kept]

        rows = 0
        for row in reader:
            rows += 1
            if rows <= MAX_ROWS_SCANNED:
                for index, column in enumerate(columns):
                    column.observe(row[index] if index < len(row) else "")

    return _assemble(
        fmt="csv" if delimiter == "," else "tsv",
        rows=rows,
        scanned=min(rows, MAX_ROWS_SCANNED),
        total_columns=len(names),
        columns=[column.to_json() for column in columns],
    )


def _profile_parquet(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise AttachmentError(
            "parquet needs pyarrow in the server's environment: "
            "uv sync --extra ui  (or attach a CSV instead)"
        ) from exc

    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise AttachmentError(f"{path.name} is not readable as parquet: {exc}") from exc

    schema = parquet.schema_arrow
    names = _dedupe(list(schema.names))
    kept = names[:MAX_COLUMNS]
    columns = [_ColumnStats(name) for name in kept]
    rows = parquet.metadata.num_rows if parquet.metadata else 0

    scanned = 0
    for batch in parquet.iter_batches(batch_size=8192, columns=kept):
        as_dict = batch.to_pydict()
        height = batch.num_rows
        for column in columns:
            for value in as_dict.get(column.name, []):
                column.observe("" if value is None else str(value))
        scanned += height
        if scanned >= MAX_ROWS_SCANNED:
            break

    return _assemble(
        fmt="parquet",
        rows=rows,
        scanned=min(scanned, rows or scanned),
        total_columns=len(names),
        columns=[column.to_json() for column in columns],
    )


def _assemble(
    *,
    fmt: str,
    rows: int,
    scanned: int,
    total_columns: int,
    columns: list[dict[str, Any]],
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "format": fmt,
        "rows": rows,
        "columns": columns,
    }
    if scanned < rows:
        # Say it in the profile itself. GOD reads this, and "these numbers
        # describe the first 50k rows" is the difference between a correct
        # inference and a confident wrong one.
        profile["stats_from_first_rows"] = scanned
    if total_columns > len(columns):
        profile["columns_omitted"] = total_columns - len(columns)
    return profile


class _ColumnStats:
    """One column, summarised in a single pass.

    Distinct values are collected only up to the categorical threshold. Past it
    the set is dropped and the column is reported by range: keeping every
    distinct order id would inflate the profile without making the seal or the
    projection any better.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.nulls = 0
        self.distinct: set[str] | None = set()
        self.numeric = True
        self.boolean = True
        self.low = math.inf
        self.high = -math.inf
        self.total = 0.0
        self.numeric_count = 0

    def observe(self, raw: str) -> None:
        text = raw.strip()
        if not text or text.lower() in {"na", "nan", "null", "none"}:
            self.nulls += 1
            return
        self.count += 1

        if self.distinct is not None:
            self.distinct.add(text)
            if len(self.distinct) > CATEGORICAL_MAX_UNIQUE:
                self.distinct = None

        lowered = text.lower()
        if self.boolean and lowered not in _TRUE and lowered not in _FALSE:
            self.boolean = False
        if self.numeric:
            try:
                number = float(text)
            except ValueError:
                self.numeric = False
            else:
                if math.isfinite(number):
                    self.numeric_count += 1
                    self.total += number
                    self.low = min(self.low, number)
                    self.high = max(self.high, number)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.boolean and self.count:
            out["type"] = "boolean"
        elif self.numeric and self.numeric_count:
            out["type"] = "number"
        else:
            out["type"] = "text"

        if self.nulls:
            out["nulls"] = self.nulls
        if self.distinct is not None:
            out["unique"] = len(self.distinct)
            # A CATEGORY REPEATS; AN IDENTIFIER DOES NOT. Without this, a column
            # of order ids in a small file slips under the cardinality cap and
            # every id lands in the seal -- 21 sealed terms for a five-row
            # table, most of them primary keys and ISO dates. Requiring at least
            # one repeat costs nothing on real categorical data (three regions
            # across a thousand rows) and excludes key and timestamp columns
            # exactly where the absolute cap cannot tell them apart.
            categorical = len(self.distinct) < self.count
            if out["type"] != "number" and categorical:
                # ONE STRING, not a list, and this is not cosmetic.
                # `transformer._source_units` recurses into every list it finds
                # and bills one projection obligation per element, so a list
                # here costs an obligation per category: 40 columns of 24
                # categories would demand a thousand hand-authored mappings and
                # never validate. Joined, a column is exactly one unit.
                out["values"] = VALUE_SEPARATOR.join(
                    sorted(self.distinct)[:MAX_SAMPLE_VALUES]
                )
        if out["type"] == "number" and self.numeric_count:
            out["min"] = _trim(self.low)
            out["max"] = _trim(self.high)
            out["mean"] = _trim(self.total / self.numeric_count)
        return out


# -- the seal --------------------------------------------------------------
def sealed_terms(profile: dict[str, Any]) -> list[str]:
    """Native vocabulary to hide, taken from the table's own schema.

    Column names and the values of low-cardinality columns, minus names too
    generic to identify anything (`_GENERIC_COLUMNS`). Numeric and boolean
    columns contribute their name but never their values -- a seal containing
    "6000.0", or "true", would rewrite every unrelated number and every plain
    English "true" in the problem statement.

    Mechanical on purpose. `build_problem` refuses to guess sealed terms from
    prose because a wrong guess reports a run as sealed against words the user
    never named; a column header is not a guess, it is the schema the user
    handed us.
    """

    terms: list[str] = []
    for column in profile.get("columns") or []:
        name = str(column.get("name") or "").strip()
        if name and name.lower() not in _GENERIC_COLUMNS:
            terms.append(name)
            # `region` may be generic while `west`/`east` are not, so values are
            # considered regardless of whether their column name survived.
        if column.get("type") in {"number", "boolean"}:
            continue
        for value in str(column.get("values") or "").split(VALUE_SEPARATOR):
            text = value.strip()
            if text and not _looks_numeric(text):
                terms.append(text)

    # `native_terms` drops anything under three characters and de-duplicates
    # anyway; doing it here too keeps what the UI shows equal to what is sealed.
    seen: dict[str, None] = {}
    for term in terms:
        if len(term) >= 3:
            seen.setdefault(term, None)
    return list(seen)


# -- helpers ---------------------------------------------------------------
def _looks_numeric(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _trim(value: float) -> float | int:
    if value == int(value) and abs(value) < 1e15:
        return int(value)
    return round(value, 6)


def _dedupe(names: list[str]) -> list[str]:
    """Column names are dict keys downstream; duplicates would silently merge."""

    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _safe_name(name: str) -> str:
    """A browser-supplied filename becomes a path under a Modal volume."""

    base = Path(str(name or "").replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    if not cleaned:
        raise AttachmentError("attachment has no usable filename")
    return cleaned[:96]


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size}B"
