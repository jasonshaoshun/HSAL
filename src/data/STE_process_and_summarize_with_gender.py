#!/usr/bin/env python3

import argparse
import contextlib
import io
import json
import re
import unicodedata
import warnings
from pathlib import Path
from typing import List, Optional

import country_converter as coco
import emoji
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from ftfy import fix_text
from unidecode import unidecode


cc = coco.CountryConverter()

CODE_BLOCK_RE = re.compile(r"<pre\b[^>]*>.*?</pre>", flags=re.DOTALL | re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"<code\b[^>]*>.*?</code>", flags=re.DOTALL | re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
ONLY_PUNCT_RE = re.compile(r"^[\s\W_]+$")
ISO3_RE = re.compile(r"^[A-Z]{3}$")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"(?i)\b(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}\b")
DISPLAY_NAME_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*", flags=re.UNICODE)
HTML_TAG_RE = re.compile(r"<[a-zA-Z][^>]*>")
URL_LIKE_RE = re.compile(r"^\s*(https?://|www\.)", re.IGNORECASE)

VALID_ISO3 = {
    code
    for code in cc.data["ISO3"].dropna().astype(str).str.upper().str.strip()
    if ISO3_RE.fullmatch(code)
}

ALIASES_TO_ISO3 = {
    "usa": "USA",
    "u.s.a.": "USA",
    "u.s.": "USA",
    "us": "USA",
    "united states": "USA",
    "america": "USA",
    "uk": "GBR",
    "u.k.": "GBR",
    "united kingdom": "GBR",
    "great britain": "GBR",
    "britain": "GBR",
    "england": "GBR",
    "scotland": "GBR",
    "wales": "GBR",
    "northern ireland": "GBR",
    "deutschland": "DEU",
    "österreich": "AUT",
    "osterreich": "AUT",
    "españa": "ESP",
    "espana": "ESP",
    "méxico": "MEX",
    "mexico": "MEX",
    "brasil": "BRA",
    "россия": "RUS",
    "rossiia": "RUS",
    "russia": "RUS",
    "中国": "CHN",
    "中华人民共和国": "CHN",
    "대한민국": "KOR",
    "日本": "JPN",
    "italia": "ITA",
    "polska": "POL",
    "nederland": "NLD",
    "schweiz": "CHE",
    "suisse": "CHE",
    "svizzera": "CHE",
    "sverige": "SWE",
    "uae": "ARE",
    "czech republic": "CZE",
    "czechia": "CZE",
    "viet nam": "VNM",
    "vietnam": "VNM",
    "hong kong": "HKG",
    "taiwan": "TWN",
}

TLD_TO_ISO3 = {
    "us": "USA",
    "gb": "GBR",
    "uk": "GBR",
    "de": "DEU",
    "fr": "FRA",
    "es": "ESP",
    "it": "ITA",
    "nl": "NLD",
    "br": "BRA",
    "ru": "RUS",
    "cn": "CHN",
    "in": "IND",
    "au": "AUS",
    "ca": "CAN",
    "se": "SWE",
    "ch": "CHE",
    "jp": "JPN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean STE raw jsonl, infer gender, cap per-user posts, and export user summary statistics."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--cleaned_output", type=str, required=True)
    parser.add_argument("--figdir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--summary_output_jsonl", type=str, required=True)
    parser.add_argument("--histogram_jsonl", type=str, required=True)
    parser.add_argument("--histogram_png", type=str, required=True)
    parser.add_argument("--boy_names_path", type=str, default="data/name_list_by_gender/boy.jsonl")
    parser.add_argument("--girl_names_path", type=str, default="data/name_list_by_gender/girl.jsonl")
    parser.add_argument("--max_posts_per_user", type=int, default=100)
    parser.add_argument("--bin_width", type=int, default=100)
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def strip_code_blocks(html: Optional[str]) -> str:
    if not isinstance(html, str):
        return ""
    without_pre = CODE_BLOCK_RE.sub(" ", html)
    without_code = INLINE_CODE_RE.sub(" ", without_pre)
    return without_code


def strip_html(html: Optional[str]) -> str:
    if not isinstance(html, str):
        return ""

    html = html.strip()
    if not html:
        return ""

    # Avoid BeautifulSoup warnings for plain URLs or plain text.
    if URL_LIKE_RE.match(html) or not HTML_TAG_RE.search(html):
        return WHITESPACE_RE.sub(" ", html).strip()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
        soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text(separator=" ")
    return WHITESPACE_RE.sub(" ", text).strip()


def _word_count(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return len([w for w in s.strip().split() if w])


def norm(raw: Optional[str]) -> str:
    if not isinstance(raw, str):
        return ""
    s = fix_text(raw).strip()
    return WHITESPACE_RE.sub(" ", s)


def plain(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = emoji.replace_emoji(s, replace="")
    s = unidecode(s)
    s = (
        s.replace("/", " ")
        .replace("|", " ")
        .replace("-", " ")
        .replace("–", " ")
        .replace("—", " ")
    )
    s = re.sub(r"[()]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def looks_like_ip(s_plain: str) -> bool:
    return bool(IPV4_RE.search(s_plain) or IPV6_RE.search(s_plain))


def tld_hint_iso3(s_plain: str) -> Optional[str]:
    parts = s_plain.split(".")
    if len(parts) < 2:
        return None
    return TLD_TO_ISO3.get(parts[-1])


def valid_iso3_or_none(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    iso3 = value.strip().upper()
    if ISO3_RE.fullmatch(iso3) and iso3 in VALID_ISO3:
        return iso3
    return None


def normalize_country_universal(value: Optional[str]) -> str:
    return valid_iso3_or_none(value) or "UNKNOWN"


def _convert_name_to_iso3(name: str) -> Optional[str]:
    if not name:
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res = cc.convert(names=[name], to="ISO3", not_found=None)
        iso3 = res[0] if isinstance(res, list) else res
        return valid_iso3_or_none(iso3)
    except Exception:
        return None


def infer_country_iso3(raw: Optional[str]) -> str:
    s = norm(raw)
    s_plain = plain(s)

    if not s_plain or ONLY_PUNCT_RE.match(s_plain) or looks_like_ip(s_plain):
        return "UNKNOWN"

    if s_plain in ALIASES_TO_ISO3:
        return ALIASES_TO_ISO3[s_plain]

    def token_to_iso3(token: str) -> Optional[str]:
        t = plain(token)
        if not t or ONLY_PUNCT_RE.match(t):
            return None
        if t in ALIASES_TO_ISO3:
            return ALIASES_TO_ISO3[t]
        iso3 = valid_iso3_or_none(t)
        if iso3:
            return iso3
        if re.fullmatch(r"[a-z]{2}", t):
            return None
        iso3 = _convert_name_to_iso3(t.title())
        if iso3:
            return iso3
        return None

    def scan_right_to_left(pieces: List[str]) -> Optional[str]:
        for piece in reversed([p for p in pieces if p.strip()]):
            iso3 = token_to_iso3(piece)
            if iso3:
                return iso3
        return None

    iso3 = scan_right_to_left([p.strip() for p in s.split(",")])
    if iso3:
        return iso3

    iso3 = scan_right_to_left(re.split(r"[|/–—\-()]", s))
    if iso3:
        return iso3

    iso3 = tld_hint_iso3(s_plain)
    if iso3:
        return iso3

    if re.fullmatch(r"[a-z]{2}", s_plain):
        return "UNKNOWN"

    iso3 = _convert_name_to_iso3(s_plain.title())
    if iso3:
        return iso3

    return "UNKNOWN"


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    no_code = df.get("AnswerBody", pd.Series(dtype=object)).apply(strip_code_blocks)
    df["AnswerBodyClean"] = no_code.apply(strip_html)
    df["AboutMeClean"] = df.get("AboutMe", pd.Series(dtype=object)).apply(strip_html)

    loc_series = df.get("Location", pd.Series(dtype=object)).fillna("").astype(str)
    df["CountryUniversal"] = loc_series.apply(infer_country_iso3)

    title_series = df.get("QuestionTitle", pd.Series(dtype=object)).fillna("").astype(str)
    df["TextClean"] = (
        "Question: "
        + title_series.fillna("").astype(str)
        + ", Answer: "
        + df["AnswerBodyClean"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return df


def _sanitize_mpl_text(s):
    if not isinstance(s, str):
        s = str(s)
    return s.replace("$", r"\$").replace("%", r"\%")


def remove_surrogates(value):
    if isinstance(value, str):
        return value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return value


def sanitize_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        out[col] = out[col].map(remove_surrogates)
    return out


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    ensure_parent_dir(path)
    safe_df = sanitize_dataframe_for_json(df)
    safe_df.to_json(path, orient="records", lines=True, force_ascii=False)


def process_types_for_saving(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()

    if "CountryUniversal" in df_out.columns:
        df_out["CountryUniversal"] = df_out["CountryUniversal"].apply(normalize_country_universal)

    for col in ("Reputation", "AnswerScore", "UserUpVotes", "UserDownVotes", "Engagement"):
        if col in df_out.columns:
            s = pd.to_numeric(df_out[col], errors="coerce")
            s_rounded = s.apply(lambda x: int(np.ceil(x)) if pd.notna(x) else pd.NA).astype("Int64")
            df_out[col] = s_rounded

    if "gender" in df_out.columns:
        df_out["gender"] = df_out["gender"].fillna("UNKNOWN")

    return df_out


def print_distributions(df: pd.DataFrame, label: str):
    print(f"--- Distribution for {label} (n={len(df)}) ---")

    if "CountryUniversal" in df.columns:
        cu = df["CountryUniversal"].fillna("UNKNOWN").astype(str)
        print("CountryUniversal counts:\n", cu.value_counts().to_string())

    if "gender" in df.columns:
        g = df["gender"].fillna("UNKNOWN").astype(str)
        print("\ngender counts:\n", g.value_counts().to_string())

    for col in ("Reputation", "AnswerScore", "UserUpVotes", "UserDownVotes", "Engagement"):
        if col in df.columns:
            print(f"\n{col} descriptive stats:")
            s = pd.to_numeric(df[col], errors="coerce")
            print(s.describe())

    print("----------------------------------------")


def load_name_set(path: str) -> set[str]:
    names = set()

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            candidates = []

            if line[0] in "{[":
                try:
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        candidates = [obj]
                    elif isinstance(obj, list):
                        candidates = [x for x in obj if isinstance(x, str)]
                    elif isinstance(obj, dict):
                        for key in ("name", "first_name", "firstName", "value", "text"):
                            if key in obj and isinstance(obj[key], str):
                                candidates = [obj[key]]
                                break
                except json.JSONDecodeError:
                    candidates = [line]
            else:
                candidates = [line]

            for candidate in candidates:
                name = normalize_name_token(candidate)
                if name:
                    names.add(name)

    return names


def normalize_name_token(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower().strip()


def choose_display_name(names: pd.Series) -> str:
    valid_names = []
    for value in names.tolist():
        if pd.isna(value):
            continue
        name = str(value).strip()
        if name:
            valid_names.append(name)

    if not valid_names:
        return ""

    counts = {}
    first_seen = {}
    for index, name in enumerate(valid_names):
        counts[name] = counts.get(name, 0) + 1
        if name not in first_seen:
            first_seen[name] = index

    return min(first_seen.keys(), key=lambda name: (-counts[name], first_seen[name]))


def extract_candidate_name(display_name: str) -> str:
    if not display_name:
        return ""
    tokens = DISPLAY_NAME_TOKEN_RE.findall(str(display_name))
    if not tokens:
        return ""
    for token in tokens:
        normalized = normalize_name_token(token.strip("-'"))
        if len(normalized) >= 2:
            return normalized
    return ""


def predict_gender(display_name: str, boy_names: set[str], girl_names: set[str]) -> tuple[str | None, str | None]:
    candidate_name = extract_candidate_name(display_name)
    if not candidate_name:
        return None, None

    in_boy_list = candidate_name in boy_names
    in_girl_list = candidate_name in girl_names

    if in_boy_list and not in_girl_list:
        return "boy", candidate_name
    if in_girl_list and not in_boy_list:
        return "girl", candidate_name
    return None, None


def build_user_summary(df: pd.DataFrame, boy_names: set[str], girl_names: set[str]) -> pd.DataFrame:
    grouped = (
        df.groupby("UserId", sort=False)
        .agg(
            DisplayName=("DisplayName", choose_display_name),
            post_count=("UserId", "size"),
        )
        .reset_index()
    )

    grouped["UserId_numeric"] = pd.to_numeric(grouped["UserId"], errors="coerce")
    grouped = grouped.sort_values(
        by=["post_count", "UserId_numeric", "UserId"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    grouped = grouped.drop(columns=["UserId_numeric"])

    gender_predictions = grouped["DisplayName"].apply(
        predict_gender,
        boy_names=boy_names,
        girl_names=girl_names,
    )
    grouped["gender"] = gender_predictions.map(lambda item: item[0])
    grouped["gender_name_match"] = gender_predictions.map(lambda item: item[1])

    return grouped[["UserId", "DisplayName", "post_count", "gender", "gender_name_match"]]


def attach_gender_columns(df: pd.DataFrame, user_summary: pd.DataFrame) -> pd.DataFrame:
    gender_lookup = user_summary.set_index("UserId")["gender"].to_dict()
    gender_name_match_lookup = user_summary.set_index("UserId")["gender_name_match"].to_dict()

    out = df.copy()
    out["gender"] = out["UserId"].map(gender_lookup).fillna("UNKNOWN")
    out["gender_name_match"] = out["UserId"].map(gender_name_match_lookup)
    return out


def cap_posts_per_user(df: pd.DataFrame, max_posts_per_user: int) -> pd.DataFrame:
    if max_posts_per_user <= 0:
        raise ValueError("--max_posts_per_user must be positive")
    if "UserId" not in df.columns:
        raise ValueError("Missing required column: UserId")

    capped = df.copy()
    capped["_post_rank_within_user"] = capped.groupby("UserId", sort=False).cumcount()
    capped = capped[capped["_post_rank_within_user"] < max_posts_per_user].copy()
    capped = capped.drop(columns=["_post_rank_within_user"])
    return capped


def build_histogram_summary(user_summary: pd.DataFrame, bin_width: int) -> pd.DataFrame:
    if bin_width <= 0:
        raise ValueError("--bin_width must be positive")

    columns = [
        "record_type",
        "subset",
        "bin_start",
        "bin_end",
        "bin_label",
        "user_count",
        "threshold_bin_start",
        "threshold_bin_label",
        "UserId",
        "DisplayName",
        "post_count",
        "gender",
        "gender_name_match",
    ]

    if user_summary.empty:
        return pd.DataFrame(columns=columns)

    histogram = user_summary[["UserId", "DisplayName", "post_count", "gender", "gender_name_match"]].copy()
    histogram["bin_start"] = (histogram["post_count"] // bin_width) * bin_width
    histogram["bin_end"] = histogram["bin_start"] + bin_width - 1
    histogram["bin_label"] = histogram["bin_start"].astype(int).astype(str) + "-" + histogram["bin_end"].astype(int).astype(str)

    histogram_bins = (
        histogram.groupby(["bin_start", "bin_end", "bin_label"], as_index=False)
        .size()
        .rename(columns={"size": "user_count"})
        .sort_values(by=["bin_start"])
        .reset_index(drop=True)
    )
    histogram_bins["record_type"] = "histogram_bin"
    histogram_bins["subset"] = "all_users"
    histogram_bins["threshold_bin_start"] = pd.NA
    histogram_bins["threshold_bin_label"] = pd.NA
    histogram_bins["UserId"] = pd.NA
    histogram_bins["DisplayName"] = pd.NA
    histogram_bins["post_count"] = pd.NA
    histogram_bins["gender"] = pd.NA
    histogram_bins["gender_name_match"] = pd.NA
    histogram_bins = histogram_bins[columns]

    outputs = [histogram_bins]

    first_sparse = histogram_bins[
        (histogram_bins["record_type"] == "histogram_bin") &
        (pd.to_numeric(histogram_bins["user_count"], errors="coerce") < 10)
    ]

    if not first_sparse.empty:
        threshold_row = first_sparse.iloc[0]
        threshold_start = int(threshold_row["bin_start"])
        detailed_users = user_summary.loc[
            user_summary["post_count"] > threshold_start,
            ["UserId", "DisplayName", "post_count", "gender", "gender_name_match"],
        ].copy()
        detailed_users["record_type"] = "threshold_user"
        detailed_users["subset"] = "all_users"
        detailed_users["bin_start"] = pd.NA
        detailed_users["bin_end"] = pd.NA
        detailed_users["bin_label"] = pd.NA
        detailed_users["user_count"] = pd.NA
        detailed_users["threshold_bin_start"] = threshold_start
        detailed_users["threshold_bin_label"] = threshold_row["bin_label"]
        detailed_users = detailed_users[columns]
        detailed_users = detailed_users.sort_values(
            by=["post_count", "UserId"],
            ascending=[False, True],
        ).reset_index(drop=True)
        outputs.append(detailed_users)

    return pd.concat(outputs, ignore_index=True, sort=False)


def save_user_post_histogram(histogram_df: pd.DataFrame, path: str, dataset_name: str) -> None:
    ensure_parent_dir(path)

    histogram_rows = histogram_df[histogram_df["record_type"] == "histogram_bin"].copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    if histogram_rows.empty:
        ax.text(0.5, 0.5, "No users available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.bar(histogram_rows["bin_label"], histogram_rows["user_count"])
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title(f"{dataset_name}: User post count distribution (all users)")
    ax.set_xlabel("Number of posts")
    ax.set_ylabel("Number of users (log scale)")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_hist_plot(series: pd.Series, title: str, xlabel: str, path: str, bins: int = 30) -> None:
    ensure_parent_dir(path)
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    if numeric.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(numeric, bins=bins)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_bar_plot_from_counts(
    counts: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: str,
    rotation: int = 45,
) -> None:
    ensure_parent_dir(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    if counts.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        labels = [_sanitize_mpl_text(x) for x in counts.index.astype(str)]
        ax.bar(labels, counts.values)
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_plots(df: pd.DataFrame, figdir: str, dataset_name: str, suffix: str = "", capped_label: str = "") -> None:
    Path(figdir).mkdir(parents=True, exist_ok=True)

    title_suffix = f" ({capped_label})" if capped_label else ""

    if "AnswerScore" in df.columns:
        _save_hist_plot(
            series=df["AnswerScore"],
            title=f"Distribution of AnswerScore{title_suffix}",
            xlabel="AnswerScore",
            path=str(Path(figdir) / f"answer_score_distribution{suffix}.png"),
            bins=30,
        )

    if "Reputation" in df.columns:
        _save_hist_plot(
            series=df["Reputation"],
            title=f"Distribution of Reputation{title_suffix}",
            xlabel="Reputation",
            path=str(Path(figdir) / f"reputation_distribution{suffix}.png"),
            bins=30,
        )

    if "Engagement" in df.columns:
        _save_hist_plot(
            series=df["Engagement"],
            title=f"Distribution of Engagement{title_suffix}",
            xlabel="Engagement (UpVotes + DownVotes)",
            path=str(Path(figdir) / f"engagement_distribution{suffix}.png"),
            bins=30,
        )

    if "CountryUniversal" in df.columns:
        countries = df["CountryUniversal"].fillna("UNKNOWN").astype(str)
        country_counts = countries[countries.str.upper() != "UNKNOWN"].value_counts().head(15)
        _save_bar_plot_from_counts(
            counts=country_counts,
            title=f"Top 15 Answerer Countries{title_suffix}",
            xlabel="Country (ISO-3)",
            ylabel="Count",
            path=str(Path(figdir) / f"country_distribution{suffix}.png"),
            rotation=45,
        )

    if "gender" in df.columns:
        gender_series = df["gender"].fillna("UNKNOWN").astype(str)
        ordered_counts = pd.Series(
            {
                "boy": int((gender_series == "boy").sum()),
                "girl": int((gender_series == "girl").sum()),
                "UNKNOWN": int((gender_series == "UNKNOWN").sum()),
            }
        )
        ordered_counts = ordered_counts[ordered_counts > 0]
        _save_bar_plot_from_counts(
            counts=ordered_counts,
            title=f"Distribution of Gender{title_suffix}",
            xlabel="Predicted gender",
            ylabel="Count",
            path=str(Path(figdir) / f"gender_distribution{suffix}.png"),
            rotation=0,
        )


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.input}, dataset name {args.dataset_name}")

    Path(args.cleaned_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.figdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.input, lines=True)
    df_clean = process_dataframe(df)

    boy_names = load_name_set(args.boy_names_path)
    girl_names = load_name_set(args.girl_names_path)

    # Build user summary on the uncapped data.
    user_summary = build_user_summary(df_clean, boy_names, girl_names)
    user_summary_save = user_summary.copy()
    user_summary_save["gender"] = user_summary_save["gender"].fillna("UNKNOWN")

    # Save user-level outputs.
    histogram_summary = build_histogram_summary(user_summary, args.bin_width)
    save_jsonl(user_summary_save, args.summary_output_jsonl)
    save_jsonl(histogram_summary, args.histogram_jsonl)
    save_user_post_histogram(histogram_summary, args.histogram_png, args.dataset_name)

    # Attach gender to the row-level dataframe.
    df_with_gender_uncapped = attach_gender_columns(df_clean, user_summary)

    # Save uncapped plots with *_uncapped.png
    uncapped_plot_df = process_types_for_saving(df_with_gender_uncapped)
    make_plots(
        uncapped_plot_df,
        figdir=args.figdir,
        dataset_name=args.dataset_name,
        suffix="_uncapped",
        capped_label="uncapped",
    )

    # Cap rows for the cleaned dataset and for the main plots.
    df_capped = cap_posts_per_user(df_with_gender_uncapped, args.max_posts_per_user)
    cleaned_save = process_types_for_saving(df_capped)

    # Save cleaned capped output.
    save_jsonl(cleaned_save, args.cleaned_output)

    # Save capped plots with the standard names.
    make_plots(
        cleaned_save,
        figdir=args.figdir,
        dataset_name=args.dataset_name,
        suffix="",
        capped_label=f"capped at {args.max_posts_per_user} posts/user",
    )

    print_distributions(cleaned_save, "cleaned_capped_all_countries")

    print(f"Saved cleaned output to {args.cleaned_output}")
    print(f"Saved user summary to {args.summary_output_jsonl}")
    print(f"Saved histogram summary to {args.histogram_jsonl}")
    print(f"Saved histogram plot to {args.histogram_png}")
    print(f"Saved capped plots to {args.figdir}/*.png")
    print(f"Saved uncapped plots to {args.figdir}/*_uncapped.png")
    print(f"Rows before cap: {len(df_with_gender_uncapped)}")
    print(f"Rows after cap: {len(cleaned_save)}")
    print(f"Users summarized: {len(user_summary_save)}")


if __name__ == "__main__":
    main()