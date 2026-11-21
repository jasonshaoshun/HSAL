#!/usr/bin/env python3
"""Summarize user retention under data_processing.py's AboutMe length filter.

The input files are read one JSON object at a time so this audit can run on the
multi-gigabyte Stack Exchange exports without loading them into memory.

By default, the script reads the four exports used by ``STE_processing.sh``::

    python src/data/summarize_aboutme_user_retention.py

Custom domain/path pairs can be supplied by repeating ``--dataset``::

    python src/data/summarize_aboutme_user_retention.py \
        --dataset "Stack Overflow=data/stackoverflow.jsonl" \
        --dataset "Mathematics=data/mathematics.jsonl" \
        --output aboutme_user_retention.md

A user is retained if at least one of their rows passes the biography criterion
in ``data_processing.filter_dataframe``: after HTML removal, the AboutMe text
must contain at least three whitespace-delimited words. Therefore,
"Missing/too-short description" counts candidate users for whom every row fails
that criterion. Recorded artifacts produced through
``STE_process_and_summarize_with_gender.py`` used a different, empty-only rule;
use ``--min-words 1`` to audit that rule.

Note that ``src/data/extract_data.sql`` currently excludes rows where AboutMe is
NULL.  To audit those upstream exclusions too, remove that SQL predicate,
re-export the candidates, and run this script on the unfiltered exports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning


DEFAULT_DATASETS = (
    ("Stack Overflow", Path("data/STE/from_website/data/stackoverflow.jsonl")),
    ("Mathematics", Path("data/STE/from_website/data/mathematics.jsonl")),
    ("Super User", Path("data/STE/from_website/data/superuser.jsonl")),
    ("English", Path("data/STE/from_website/data/english.jsonl")),
)

WHITESPACE_RE = re.compile(r"\s+")

# A biography containing only a URL is data, not a path Beautiful Soup should open.
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


@dataclass(frozen=True)
class RetentionSummary:
    domain: str
    all_candidate_users: int
    missing_or_too_short: int
    retained_users: int

    @property
    def retention_rate(self) -> float:
        if self.all_candidate_users == 0:
            return 0.0
        return 100.0 * self.retained_users / self.all_candidate_users


def parse_dataset(value: str) -> tuple[str, Path]:
    """Parse a command-line dataset specification of the form DOMAIN=PATH."""
    domain, separator, raw_path = value.partition("=")
    if not separator or not domain.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            f"invalid dataset {value!r}; expected a value such as "
            "'Stack Overflow=data/stackoverflow.jsonl'"
        )
    return domain.strip(), Path(raw_path).expanduser()


def has_sufficient_aboutme(raw_aboutme: object, min_words: int) -> bool:
    """Apply data_processing.py's AboutMe cleaning and word-count rule."""
    # Keep this identical to strip_html() + _word_count() in data_processing.py.
    if not isinstance(raw_aboutme, str):
        raw_aboutme = ""
    text = BeautifulSoup(raw_aboutme, "html.parser").get_text(separator=" ")
    clean_text = WHITESPACE_RE.sub(" ", text).strip()
    word_count = len([word for word in clean_text.split() if word])
    return word_count >= min_words


def summarize_dataset(
    domain: str,
    path: Path,
    min_words: int = 3,
    user_column: str = "UserId",
    aboutme_column: str = "AboutMe",
) -> RetentionSummary:
    """Count distinct candidate and retained users in one JSONL export."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset for {domain!r} does not exist: {path}")

    # False becomes True once any row belonging to the user passes the filter.
    retained_by_user: dict[object, bool] = {}
    aboutme_eligibility_cache: dict[object, bool] = {}

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error

            if user_column not in row:
                raise ValueError(
                    f"missing required column {user_column!r} in {path} "
                    f"at line {line_number}"
                )

            user_id = row[user_column]
            if user_id is None:
                # A null ID cannot identify a distinct candidate user.
                continue

            raw_aboutme = row.get(aboutme_column)
            try:
                aboutme_ok = aboutme_eligibility_cache[raw_aboutme]
            except (KeyError, TypeError):
                aboutme_ok = has_sufficient_aboutme(raw_aboutme, min_words)
                try:
                    aboutme_eligibility_cache[raw_aboutme] = aboutme_ok
                except TypeError:
                    # AboutMe should be a string or null, but remain correct for
                    # malformed, unhashable JSON values rather than caching them.
                    pass

            retained_by_user[user_id] = (
                retained_by_user.get(user_id, False) or aboutme_ok
            )

    all_candidate_users = len(retained_by_user)
    retained_users = sum(retained_by_user.values())
    return RetentionSummary(
        domain=domain,
        all_candidate_users=all_candidate_users,
        missing_or_too_short=all_candidate_users - retained_users,
        retained_users=retained_users,
    )


def render_markdown(summaries: Iterable[RetentionSummary]) -> str:
    """Render summaries as a right-aligned Markdown table."""
    lines = [
        "| Domain | All candidate users | Missing/too-short description "
        "| Retained users | Retention rate |",
        "|:--|--:|--:|--:|--:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.domain} "
            f"| {summary.all_candidate_users:,} "
            f"| {summary.missing_or_too_short:,} "
            f"| {summary.retained_users:,} "
            f"| {summary.retention_rate:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count users retained by an AboutMe length filter."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="DOMAIN=PATH",
        type=parse_dataset,
        help="domain label and input JSONL path; repeat for multiple domains",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="minimum words required after stripping AboutMe HTML (default: 3)",
    )
    parser.add_argument(
        "--user-column",
        default="UserId",
        help="JSON field containing the user identifier (default: UserId)",
    )
    parser.add_argument(
        "--aboutme-column",
        default="AboutMe",
        help="JSON field containing the raw biography (default: AboutMe)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the Markdown table to this file instead of standard output",
    )
    args = parser.parse_args(argv)
    if args.min_words < 0:
        parser.error("--min-words must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = args.dataset or DEFAULT_DATASETS
    summaries = [
        summarize_dataset(
            domain,
            path,
            min_words=args.min_words,
            user_column=args.user_column,
            aboutme_column=args.aboutme_column,
        )
        for domain, path in datasets
    ]
    table = render_markdown(summaries)

    if args.output is None:
        sys.stdout.write(table)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
