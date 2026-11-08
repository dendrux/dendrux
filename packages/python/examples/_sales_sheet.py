"""Small local CSV adapter for one sales-demo operator at a time."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from tempfile import NamedTemporaryFile

from dendrux import tool

_COLUMNS = ["order", "region", "amount"]
_INITIAL = "order,region,amount\nA1, north ,100\nA2,SOUTH,200\nA3,NORTH,50\n"


class SalesSheet:
    """Own a bounded sheet and a review inbox, independent of the agent database."""

    def __init__(self, root: Path, *, max_rows: int = 100, max_bytes: int = 32768) -> None:
        if max_rows < 1 or max_bytes < 1:
            raise ValueError("demo_invalid_sheet_limits")
        self.path = root / "sales.csv"
        self.inbox = root / "review-inbox.log"
        self.max_rows = max_rows
        self.max_bytes = max_bytes

    def initialize(self) -> None:
        """Create the three-row fixture once, leaving existing sales data intact."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8", newline="") as file:
                file.write(_INITIAL)
        except FileExistsError:
            pass

    def _read(self) -> list[dict[str, str]]:
        with self.path.open("rb") as file:
            data = file.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("demo_sheet_too_large")
        try:
            reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
            if next(reader, None) != _COLUMNS:
                raise ValueError("demo_invalid_sheet")
            rows: list[dict[str, str]] = []
            for values in reader:
                if len(rows) >= self.max_rows:
                    raise ValueError("demo_sheet_too_many_rows")
                if len(values) != len(_COLUMNS):
                    raise ValueError("demo_invalid_sheet")
                rows.append(dict(zip(_COLUMNS, values, strict=True)))
        except (UnicodeDecodeError, csv.Error):
            raise ValueError("demo_invalid_sheet") from None
        return rows

    @tool(idempotent=True, max_calls_per_run=4)
    async def read_sheet(self) -> list[dict[str, str]]:
        """Read sales rows; row numbers start at one after the header."""
        return self._read()

    @tool(parallel=False, max_calls_per_run=4)
    async def write_region(self, row_number: int, region: str) -> str:
        """Set a region cell after approval, preserving row order, order IDs and amounts."""
        rows = self._read()
        if not 1 <= row_number <= len(rows) or region not in {"NORTH", "SOUTH"}:
            raise ValueError("demo_invalid_cell_write")
        rows[row_number - 1]["region"] = region
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        data = buffer.getvalue().encode("utf-8")
        if len(data) > self.max_bytes:
            raise ValueError("demo_sheet_too_large")
        with NamedTemporaryFile(dir=self.path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            try:
                file.write(data)
                file.close()
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
        return "region updated"

    @tool(parallel=False, max_calls_per_run=1)
    async def post_review_notice(self) -> str:
        """Append a review-ready notice to the analyst inbox."""
        with self.inbox.open("a", encoding="utf-8") as file:
            file.write("Sales cleanup ready for review\n")
        return "notice posted"
