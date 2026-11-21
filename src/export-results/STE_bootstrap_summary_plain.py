#!/usr/bin/env python3
"""Generate plain numeric STE bootstrap tables for OpenReview rebuttals."""

from STE_bootstrap_summary import run_main


if __name__ == "__main__":
    run_main(
        include_comparisons=False,
        default_out_dir="results_bootstrap/summary_textual_erasure/STE_tables_bootstrap_summary_plain",
    )
