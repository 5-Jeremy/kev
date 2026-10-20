#!/usr/bin/env python3
"""Write synthetic Kev records for a local fine-tuning dry run.

Each record has one noul, one choice, and one score question. The labels follow
directly from the ticket's response window, issue, and service status.

    python my_scripts/generate_dry_run_data.py --output /tmp/kev-dry-run.jsonl

Set data.path in the YAML for my_scripts/finetune_local.py to that output path.
The local trainer splits the file into train, calibration, and development sets.
"""

import argparse
import itertools
import json
import random
from pathlib import Path


ISSUES = {
    "billing": "The customer reports a duplicate charge on an invoice.",
    "account": "The customer cannot update their account email address.",
    "technical": "The customer reports an error while opening the application.",
}
STATUSES = (
    "The service works; this is a routine request.",
    "The service works only intermittently.",
    "The service is unavailable.",
)
SCORE_LEVELS = (
    "Routine: the service works normally",
    "Impaired: the service works intermittently",
    "Unavailable: the service does not work",
)


def records(repeats: int, seed: int):
    combinations = list(itertools.product((False, True), ISSUES, range(len(STATUSES))))
    combinations *= repeats
    random.Random(seed).shuffle(combinations)
    for number, (same_day, issue, severity) in enumerate(combinations, start=1):
        yield {
            "state": {
                "ticket": f"T-{number:04d}",
                "response_window": "same day" if same_day else "standard",
                "issue": ISSUES[issue],
                "service_status": STATUSES[severity],
            },
            "questions": {
                "same_day": {
                    "type": "noul",
                    "instructions": "Does this ticket require a same-day response?",
                    "src": "synthetic_same_day",
                    "label": same_day,
                },
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {
                        "billing": "Charges and invoices",
                        "account": "Account settings",
                        "technical": "Application errors",
                    },
                    "src": "synthetic_team",
                    "label": issue,
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is the service problem?",
                    "criteria": list(SCORE_LEVELS),
                    "src": "synthetic_severity",
                    "label": severity,
                },
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="path to the output JSONL file")
    parser.add_argument("--repeats", type=int, default=6, help="examples per label combination (default: 6; 108 records)")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed (default: 0)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records(args.repeats, args.seed):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {18 * args.repeats} records to {args.output}")


if __name__ == "__main__":
    main()
