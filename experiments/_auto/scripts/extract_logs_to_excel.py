#!/usr/bin/env python3
"""Collect test results from a folder of log files into a single comparison table.

Usage:
    python extract_test_results.py /path/to/logs
    python extract_test_results.py /path/to/logs -o results.xlsx --top-algorithms 5

Conventions:
    * File names look like ``[NO_CROP_]<...><train size>_<model name>.out``, e.g.
      ``NO_CROP_7class10k_ConvNeXtBase.out``. The model name is the last "_" part of the
      name and the train size is the number at the end of the part just before it
      (10k above). If no size is found, DEFAULT_TRAIN_SIZE (100) is used.
    * Files whose name starts with NO_CROP are "Before Crop"; all others are "After Crop".
    * Param # comes from the ``Total params:`` line (``159,873,023`` or ``18.1M``) and is
      shown shortened (18.1M).
    * Only the top algorithms (by F1) are listed per network / train size / crop side.
    * The vector such as ``[10 7 7 6 7 4 8]`` (spaces, no commas; found on the line after
      one ending in ``/step`` and before a line starting with ``[``) holds the number of predictions per class.
      A perfect run predicts the true class size for every class (``[7 7 7 7 7 7 7]``).
      Surplus predictions count as false positives, missing ones as false negatives,
      and sensitivity / specificity are macro-averaged over the classes.
    * F1 is the mean of the per-class ``Score: [...]`` values.

Sheets produced:
    Results      the comparison table (Before Crop / After Crop / PYOD Before / PYOD After)
    Runs         one row per file, for auditing the numbers
    Algorithms   every PYOD algorithm per file (not limited to the top ones)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from statistics import mean

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
NO_CROP_PREFIX = "NOCROP"

PARAMS_RE = re.compile(r"^\s*Total params:\s*([\d,]*\.?\d+)\s?([KMBkmb])?\b", re.M)
PARAM_UNITS = {"k": 1e3, "m": 1e6, "b": 1e9}
# The confusion vector sits between a line ending in "/step" and a line starting with "[":
#     ... 3s/step
#     [10  7  7  6  7  4  8]      <- numbers separated by spaces only, no commas
#     [
# Groups: step time value, its unit (ms / s), the vector.
CONFUSION_RE = re.compile(
    r"([\d.]+)(m?s)/step[ \t]*\n"
    r"[ \t]*\[\s*(\d+(?:\s+\d+)+)\s*\][ \t]*\n"
    r"[ \t]*\["
)
SCORE_RE = re.compile(r"Score:\s*\[([^\]]*)\]")
ALGO_HEADER_RE = re.compile(r"^-+\s*Running\s+(.+?)\s*-+\s*$", re.M)
ALGO_ACCURACY_RE = re.compile(r"accuracy for .+?:\s*(-?[\d.]+)")
ALGO_F1_RE = re.compile(r"f1_score:\s*(-?[\d.]+)")
TRAIN_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?[kK]?)$")
DEFAULT_TRAIN_SIZE = 100
TOP_ALGORITHMS = 3


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Algorithm:
    name: str
    accuracy: float | None
    f1: float | None


@dataclass
class Metrics:
    """All values in percent."""
    specificity: float | None
    sensitivity: float | None
    f1: float | None

    def as_cells(self) -> list:
        return [self.specificity, self.sensitivity, self.f1]


@dataclass
class Run:
    file: str
    network: str
    train_size: int
    cropped: bool
    params: int | None
    step_time_ms: float | None
    confusion: list[int]
    class_scores: list[float]
    algorithms: list[Algorithm]


@dataclass
class Block:
    """One (network, train size) entry of the results table."""
    network: str
    train_size: int
    params: int | None
    before: Metrics | None
    after: Metrics | None
    before_algorithms: list[Algorithm]
    after_algorithms: list[Algorithm]


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def last_match(pattern: re.Pattern, text: str) -> re.Match | None:
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def parse_file_name(stem: str) -> tuple[str, int]:
    """Return (model name, train size) from a name like ``NO_CROP_7class10k_ConvNeXtBase``."""
    *prefix, network = stem.split("_")
    match = TRAIN_SIZE_RE.search(prefix[-1]) if prefix else None
    if not match:
        return network, DEFAULT_TRAIN_SIZE
    token = match.group(1).lower()
    multiplier = 1000 if token.endswith("k") else 1
    return network, int(float(token.rstrip("k")) * multiplier)


def parse_params(text: str) -> int | None:
    match = PARAMS_RE.search(text)
    if not match:
        return None
    unit = PARAM_UNITS.get((match.group(2) or "").lower(), 1)
    return round(float(match.group(1).replace(",", "")) * unit)


def parse_algorithms(text: str) -> list[Algorithm]:
    headers = list(ALGO_HEADER_RE.finditer(text))
    algorithms = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[header.end():end]
        accuracy = ALGO_ACCURACY_RE.search(block)
        f1 = ALGO_F1_RE.search(block)
        algorithms.append(Algorithm(
            name=header.group(1).strip("[] "),
            accuracy=float(accuracy.group(1)) if accuracy else None,
            f1=float(f1.group(1)) if f1 else None,
        ))
    return algorithms


def parse_run(path: Path, text: str) -> Run:
    network, train_size = parse_file_name(path.stem)
    confusion = last_match(CONFUSION_RE, text)
    score = last_match(SCORE_RE, text)

    return Run(
        file=path.name,
        network=network,
        train_size=train_size,
        cropped=not path.name.upper().startswith(NO_CROP_PREFIX),
        params=parse_params(text),
        step_time_ms=(float(confusion.group(1)) * (1000 if confusion.group(2) == "s" else 1)) if confusion else None,
        confusion=[int(n) for n in confusion.group(3).split()] if confusion else [],
        class_scores=[float(v) for v in score.group(1).split()] if score else [],
        algorithms=parse_algorithms(text),
    )


def load_runs(folder: Path, pattern: str, exclude: Path) -> list[Run]:
    runs = []
    for path in sorted(folder.glob(pattern)):
        if not path.is_file() or path.resolve() == exclude:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warn(f"{path.name}: not a text file, skipped")
            continue
        runs.append(parse_run(path, text))
    return runs


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def sensitivity_specificity(
    predicted_counts: list[int], samples_per_class: float | None = None
) -> tuple[float | None, float | None]:
    """Macro-averaged (specificity, sensitivity) in percent from per-class prediction counts."""
    if not predicted_counts:
        return None, None

    total = sum(predicted_counts)
    expected = samples_per_class or total / len(predicted_counts)
    negatives = total - expected
    if expected <= 0 or negatives <= 0:
        return None, None

    specificities, sensitivities = [], []
    for predicted in predicted_counts:
        false_neg = max(expected - predicted, 0)
        false_pos = max(predicted - expected, 0)
        true_pos = expected - false_neg
        true_neg = negatives - false_pos
        sensitivities.append(true_pos / (true_pos + false_neg))
        specificities.append(true_neg / (true_neg + false_pos))
    return mean(specificities) * 100, mean(sensitivities) * 100


def run_metrics(run: Run, samples_per_class: float | None) -> Metrics:
    specificity, sensitivity = sensitivity_specificity(run.confusion, samples_per_class)
    f1 = mean(run.class_scores) * 100 if run.class_scores else None
    return Metrics(specificity, sensitivity, f1)


# --------------------------------------------------------------------------- #
# Table assembly
# --------------------------------------------------------------------------- #
def build_blocks(runs: list[Run], samples_per_class: float | None, top_n: int) -> list[Block]:
    groups: dict[tuple, dict[str, Run]] = defaultdict(dict)
    for run in runs:
        slot = "after" if run.cropped else "before"
        key = (run.network, run.train_size)
        if slot in groups[key]:
            warn(f"{run.file} replaces {groups[key][slot].file} ({slot} crop, {key[0]}, {key[1]})")
        groups[key][slot] = run

    blocks = []
    for key in sorted(groups, key=lambda k: (k[0].lower(), k[1])):
        pair = groups[key]
        before, after = pair.get("before"), pair.get("after")
        present = [r for r in (after, before) if r]
        blocks.append(Block(
            network=key[0],
            train_size=key[1],
            params=next((r.params for r in present if r.params), None),
            before=run_metrics(before, samples_per_class) if before else None,
            after=run_metrics(after, samples_per_class) if after else None,
            before_algorithms=top_algorithms(before.algorithms, top_n) if before else [],
            after_algorithms=top_algorithms(after.algorithms, top_n) if after else [],
        ))
    return blocks


def top_algorithms(algorithms: list[Algorithm], limit: int) -> list[Algorithm]:
    """Best `limit` algorithms by F1, highest first."""
    ranked = sorted(algorithms, key=lambda a: -1 if a.f1 is None else a.f1, reverse=True)
    return ranked[:limit]


def format_size(size: int) -> str:
    return f"{size // 1000}k" if size >= 1000 and size % 1000 == 0 else str(size)


def format_params(count: int | None) -> str | None:
    """159873023 -> '159.9M', 18100000 -> '18.1M'."""
    if count is None:
        return None
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if count >= threshold:
            return f"{count / threshold:.1f}{suffix}"
    return str(count)


# --------------------------------------------------------------------------- #
# Excel output
# --------------------------------------------------------------------------- #
FONT_NAME = "Arial"
NAVY, LIGHT_BLUE = "1F3864", "D9E1F2"
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

RESULT_HEADERS = [
    "Network", "Train Size", "Param #",
    "Specificity", "Sensitivity", "F1",
    "Specificity", "Sensitivity", "F1",
    "Algorithm", "F1",
    "Algorithm", "F1",
]
GROUP_HEADERS = [
    ("Before Crop", 4, 6), ("After Crop", 7, 9),
    ("PYOD Before Crop", 10, 11), ("PYOD After Crop", 12, 13),
]
HEADER_ROWS = 2
PARAMS_COL = 3
PERCENT_COLS = (4, 5, 6, 7, 8, 9, 11, 13)
COLUMN_WIDTHS = [20, 12, 15, 13, 13, 10, 13, 13, 10, 16, 10, 16, 10]


def write_results_sheet(ws, blocks: list[Block]) -> None:
    merges = [(1, 1, 1, PARAMS_COL)]
    for label, start, end in GROUP_HEADERS:
        ws.cell(1, start, label)
        merges.append((1, start, 1, end))
    for col, header in enumerate(RESULT_HEADERS, start=1):
        ws.cell(2, col, header)

    row = HEADER_ROWS + 1
    network_start, previous_network = row, None
    for block in blocks:
        algorithm_rows = list(zip_longest(block.before_algorithms, block.after_algorithms))
        first = row
        last = row + max(len(algorithm_rows), 1) - 1

        if block.network != previous_network:
            if previous_network is not None:
                merges.append((network_start, 1, first - 1, 1))
            ws.cell(first, 1, block.network)
            network_start, previous_network = first, block.network

        values = [
            format_size(block.train_size),
            format_params(block.params),
            *(block.before.as_cells() if block.before else [None] * 3),
            *(block.after.as_cells() if block.after else [None] * 3),
        ]
        for col, value in enumerate(values, start=2):
            ws.cell(first, col, value)
            merges.append((first, col, last, col))

        for offset, (before_alg, after_alg) in enumerate(algorithm_rows):
            for algorithm, name_col in ((before_alg, 10), (after_alg, 12)):
                if algorithm:
                    ws.cell(first + offset, name_col, algorithm.name)
                    ws.cell(first + offset, name_col + 1, algorithm.f1)
        row = last + 1
    merges.append((network_start, 1, row - 1, 1))

    style_results_sheet(ws, last_row=row - 1)
    for first_row, first_col, last_row, last_col in merges:
        if (first_row, first_col) != (last_row, last_col):
            ws.merge_cells(start_row=first_row, start_column=first_col,
                           end_row=last_row, end_column=last_col)
    ws.freeze_panes = "D3"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def style_results_sheet(ws, last_row: int) -> None:
    for row in ws.iter_rows(min_row=1, max_row=last_row, max_col=len(RESULT_HEADERS)):
        for cell in row:
            cell.font = Font(name=FONT_NAME)
            cell.border = BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for cell in ws[1]:
        cell.font = Font(name=FONT_NAME, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=NAVY)
    for cell in ws[2]:
        cell.font = Font(name=FONT_NAME, bold=True)
        cell.fill = PatternFill("solid", fgColor=LIGHT_BLUE)

    for row in range(HEADER_ROWS + 1, last_row + 1):
        for col in PERCENT_COLS:
            ws.cell(row, col).number_format = "0.00"

    for col, width in enumerate(COLUMN_WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(col)].width = width


def write_plain_sheet(ws, headers: list[str], rows: list[list]) -> None:
    ws.append(headers)
    for row in rows:
        ws.append(row)

    for cell in ws[1]:
        cell.font = Font(name=FONT_NAME, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name=FONT_NAME)
            if isinstance(cell.value, int):
                cell.number_format = "#,##0"
            elif isinstance(cell.value, float):
                cell.number_format = "0.00"

    for index, column in enumerate(ws.columns, start=1):
        longest = max(len(str(c.value)) if c.value is not None else 0 for c in column)
        ws.column_dimensions[get_column_letter(index)].width = min(max(longest + 2, 12), 40)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def write_workbook(runs: list[Run], blocks: list[Block], samples_per_class: float | None,
                   output: Path) -> None:
    wb = Workbook()
    results = wb.active
    results.title = "Results"
    write_results_sheet(results, blocks)

    crop_label = lambda run: "After" if run.cropped else "Before"  # noqa: E731

    run_rows = []
    for run in runs:
        metrics = run_metrics(run, samples_per_class)
        run_rows.append([
            run.file, run.network, run.train_size, crop_label(run), run.params,
            run.step_time_ms, " ".join(map(str, run.confusion)),
            metrics.specificity, metrics.sensitivity, metrics.f1,
        ])
    write_plain_sheet(
        wb.create_sheet("Runs"),
        ["File", "Network", "Train Size", "Crop", "Param #", "Step Time (ms)",
         "Confusion Vector", "Specificity", "Sensitivity", "F1"],
        run_rows,
    )

    algorithm_rows = [
        [run.file, run.network, run.train_size, crop_label(run), a.name, a.accuracy, a.f1]
        for run in runs for a in run.algorithms
    ]
    write_plain_sheet(
        wb.create_sheet("Algorithms"),
        ["File", "Network", "Train Size", "Crop", "Algorithm", "Accuracy", "F1"],
        algorithm_rows,
    )
    wb.save(output)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder containing the result files")
    parser.add_argument("-o", "--output", type=Path, default=Path("test_results.xlsx"),
                        help="Output .xlsx path (default: ./test_results.xlsx)")
    parser.add_argument("-p", "--pattern", default="*.out", help="Glob pattern for input files (default: *.out)")
    parser.add_argument("--top-algorithms", type=int, default=TOP_ALGORITHMS,
                        help=f"Algorithms listed per network/size/crop (default: {TOP_ALGORITHMS})")
    parser.add_argument("--samples-per-class", type=float, default=None,
                        help="True samples per class (default: total predictions / number of classes)")
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"Not a directory: {folder}")
    output = args.output.expanduser().resolve()

    runs = load_runs(folder, args.pattern, exclude=output)
    if not runs:
        sys.exit(f"No usable files found in {folder} (pattern: {args.pattern})")

    blocks = build_blocks(runs, args.samples_per_class, args.top_algorithms)
    write_workbook(runs, blocks, args.samples_per_class, output)
    print(f"Processed {len(runs)} file(s), {len(blocks)} network/size combination(s) -> {output}")


if __name__ == "__main__":
    main()