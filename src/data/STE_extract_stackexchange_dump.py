#!/usr/bin/env python3
"""Extract answer-level examples from Stack Exchange XML data dumps.

The output schema matches the raw JSONL expected by
``src/data/STE_process_and_summarize_with_gender.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterator
import xml.etree.ElementTree as ET


DEFAULT_SITE_BY_DATASET = {
    "english": "english.stackexchange.com",
    "mathematics": "math.stackexchange.com",
    "stackoverflow": "stackoverflow.com",
    "superuser": "superuser.com",
}


def safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value.split(".")[0], "%Y-%m-%dT%H:%M:%S")


def parse_optional_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    return parse_datetime(value)


def iter_rows(xml_path: Path) -> Iterator[ET.Element]:
    context = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event == "end" and elem.tag == "row":
            yield elem
            elem.clear()
            root.clear()


def resolve_xml_paths(input_dir: Path) -> tuple[Path, Path]:
    direct_users = input_dir / "Users.xml"
    direct_posts = input_dir / "Posts.xml"
    if direct_users.exists() and direct_posts.exists():
        return direct_users, direct_posts

    users_files = sorted(input_dir.rglob("Users.xml"))
    posts_by_parent = {path.parent: path for path in sorted(input_dir.rglob("Posts.xml"))}
    for users_path in users_files:
        posts_path = posts_by_parent.get(users_path.parent)
        if posts_path is not None:
            return users_path, posts_path

    raise FileNotFoundError(
        f"Could not find Users.xml and Posts.xml under {input_dir}. "
        "Pass --input-dir pointing at an extracted Stack Exchange dump directory."
    )


def extract_archive(archive: Path, extract_dir: Path) -> Path:
    if shutil.which("7z") is None:
        raise RuntimeError("The --archive option requires the `7z` executable on PATH.")

    extract_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["7z", "x", "-y", f"-o{extract_dir}", str(archive)]
    print("Extracting archive:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return extract_dir


def load_users(users_file: Path) -> dict[str, dict[str, object]]:
    print(f"Step 1: Loading users with AboutMe from {users_file}...", flush=True)
    valid_users: dict[str, dict[str, object]] = {}

    for elem in iter_rows(users_file):
        about_me = elem.get("AboutMe")
        user_id = elem.get("Id")
        if not about_me or not user_id:
            continue

        up_votes = safe_int(elem.get("UpVotes"))
        down_votes = safe_int(elem.get("DownVotes"))
        valid_users[user_id] = {
            "DisplayName": elem.get("DisplayName"),
            "Location": elem.get("Location"),
            "AboutMe": about_me,
            "Reputation": elem.get("Reputation"),
            "UpVotes": up_votes,
            "DownVotes": down_votes,
            "Engagement": up_votes + down_votes,
        }

    print(f"Loaded {len(valid_users)} valid users.", flush=True)
    return valid_users


def should_keep_answer(
    elem: ET.Element,
    start_date: datetime | None,
    end_date: datetime | None,
    keep_code_posts: bool,
    include_community_owned: bool,
) -> bool:
    if not include_community_owned and elem.get("CommunityOwnedDate") is not None:
        return False

    creation_date = parse_datetime(elem.get("CreationDate"))
    if start_date is not None and (creation_date is None or creation_date < start_date):
        return False
    if end_date is not None and (creation_date is None or creation_date >= end_date):
        return False

    if not keep_code_posts:
        body = elem.get("Body", "")
        if "<pre" in body or "<code" in body:
            return False

    return True


def build_row(
    answer_elem: ET.Element,
    question_data: dict[str, str | None],
    user_data: dict[str, object],
) -> dict[str, object]:
    parent_id = answer_elem.get("ParentId")
    owner_id = answer_elem.get("OwnerUserId")
    return {
        "AnswerId": answer_elem.get("Id"),
        "QuestionId": parent_id,
        "QuestionTitle": question_data.get("Title"),
        "QuestionTags": question_data.get("Tags"),
        "QuestionCreationDate": question_data.get("CreationDate"),
        "AnswerBody": answer_elem.get("Body", ""),
        "AnswerCreationDate": answer_elem.get("CreationDate"),
        "AnswerScore": answer_elem.get("Score"),
        "QuestionScore": question_data.get("Score"),
        "QuestionViewCount": question_data.get("ViewCount"),
        "QuestionAnswerCount": question_data.get("AnswerCount"),
        "AcceptedAnswerId": question_data.get("AcceptedAnswerId"),
        "UserId": owner_id,
        "DisplayName": user_data.get("DisplayName"),
        "Location": user_data.get("Location"),
        "AboutMe": user_data.get("AboutMe"),
        "Reputation": user_data.get("Reputation"),
        "UserUpVotes": user_data.get("UpVotes"),
        "UserDownVotes": user_data.get("DownVotes"),
        "Engagement": user_data.get("Engagement"),
    }


def write_examples(
    posts_file: Path,
    output_file: Path,
    valid_users: dict[str, dict[str, object]],
    start_date: datetime | None,
    end_date: datetime | None,
    max_rows: int | None,
    keep_code_posts: bool,
    include_community_owned: bool,
) -> int:
    print(f"Step 2: Processing posts from {posts_file}...", flush=True)

    question_cache: dict[str, dict[str, str | None]] = {}
    rows_written = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for elem in iter_rows(posts_file):
            post_type = elem.get("PostTypeId")

            if post_type == "1":
                question_id = elem.get("Id")
                if question_id:
                    question_cache[question_id] = {
                        "Title": elem.get("Title"),
                        "Tags": elem.get("Tags"),
                        "CreationDate": elem.get("CreationDate"),
                        "Score": elem.get("Score"),
                        "ViewCount": elem.get("ViewCount"),
                        "AnswerCount": elem.get("AnswerCount"),
                        "AcceptedAnswerId": elem.get("AcceptedAnswerId"),
                    }
                continue

            if post_type != "2":
                continue

            if not should_keep_answer(
                elem,
                start_date=start_date,
                end_date=end_date,
                keep_code_posts=keep_code_posts,
                include_community_owned=include_community_owned,
            ):
                continue

            parent_id = elem.get("ParentId")
            owner_id = elem.get("OwnerUserId")
            if parent_id not in question_cache or owner_id not in valid_users:
                continue

            row = build_row(elem, question_cache[parent_id], valid_users[owner_id])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            rows_written += 1
            if rows_written % 1000 == 0:
                print(f"Written {rows_written} rows...", flush=True)
            if max_rows is not None and rows_written >= max_rows:
                break

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an extracted Stack Exchange XML dump into STE raw JSONL."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Local dataset id, e.g. english, mathematics, stackoverflow, or superuser.",
    )
    parser.add_argument(
        "--site",
        default=None,
        help="Stack Exchange site name used only for logging; defaults from --dataset when known.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing Users.xml and Posts.xml. Defaults to data/STE/from_website/dumps/<dataset>.",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Optional .7z dump archive to extract before conversion.",
    )
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="Directory used when --archive is provided. Defaults to data/STE/from_website/dumps/<dataset>.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to data/STE/from_website/data/<dataset>.jsonl.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive answer start date.")
    parser.add_argument("--end-date", default=None, help="Optional exclusive answer end date.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional maximum number of rows to write.")
    parser.add_argument("--keep-code-posts", action="store_true", help="Keep answers containing <pre> or <code> blocks.")
    parser.add_argument(
        "--include-community-owned",
        action="store_true",
        help="Keep community-owned answers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = args.dataset.strip()
    site = args.site or DEFAULT_SITE_BY_DATASET.get(dataset, dataset)
    input_dir = Path(args.input_dir or f"data/STE/from_website/dumps/{dataset}")
    extract_dir = Path(args.extract_dir or f"data/STE/from_website/dumps/{dataset}")
    output_file = Path(args.output or f"data/STE/from_website/data/{dataset}.jsonl")

    if args.archive is not None:
        input_dir = extract_archive(Path(args.archive), extract_dir)

    users_file, posts_file = resolve_xml_paths(input_dir)
    start_date = parse_optional_date(args.start_date)
    end_date = parse_optional_date(args.end_date)

    print(f"Dataset: {dataset}", flush=True)
    print(f"Site: {site}", flush=True)
    print(f"Users XML: {users_file}", flush=True)
    print(f"Posts XML: {posts_file}", flush=True)
    print(f"Output JSONL: {output_file}", flush=True)

    valid_users = load_users(users_file)
    rows_written = write_examples(
        posts_file=posts_file,
        output_file=output_file,
        valid_users=valid_users,
        start_date=start_date,
        end_date=end_date,
        max_rows=args.max_rows,
        keep_code_posts=args.keep_code_posts,
        include_community_owned=args.include_community_owned,
    )
    print(f"Done. Wrote {rows_written} rows to {output_file}", flush=True)


if __name__ == "__main__":
    main()
