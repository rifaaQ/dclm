import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional


HQ_LABEL = "__label__hq"
LQ_LABEL = "__label__cc"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fastText training files for two quality classifiers: "
            "MMLU vs RefinedWeb and MMLU vs DCLM."
        )
    )
    parser.add_argument("--mmlu_input", required=True, help="Path to MMLU file or directory.")
    parser.add_argument("--refinedweb_input", required=True, help="Path to RefinedWeb file or directory.")
    parser.add_argument("--dclm_input", required=True, help="Path to DCLM file or directory.")
    parser.add_argument(
        "--output_dir",
        default="./data/fasttext",
        help="Directory for generated fastText train/valid files.",
    )
    parser.add_argument(
        "--valid_frac",
        type=float,
        default=0.1,
        help="Validation fraction for each dataset split.",
    )
    parser.add_argument(
        "--max_per_source",
        type=int,
        default=0,
        help="If >0, cap lines loaded per source before balancing.",
    )
    parser.add_argument(
        "--text_field",
        default="text",
        help="Preferred text field when reading JSONL (fallbacks are used automatically).",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=10,
        help="Drop examples shorter than this many characters after strip.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _extract_text_from_json(obj: dict, preferred_field: str) -> Optional[str]:
    if preferred_field in obj and isinstance(obj[preferred_field], str):
        return obj[preferred_field]
    for field in ("text", "content", "body", "raw_content", "document"):
        if field in obj and isinstance(obj[field], str):
            return obj[field]
    return None


def _answer_to_choice_letter(answer: object, num_choices: int) -> Optional[str]:
    if answer is None:
        return None
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        if 0 <= answer < num_choices:
            return chr(ord("A") + answer)
        return None
    if isinstance(answer, str):
        s = answer.strip()
        if not s:
            return None
        ch = s[0].upper()
        if "A" <= ch <= "Z" and ord(ch) - ord("A") < num_choices:
            return ch
    return None


def _choices_as_lines(choices: object) -> Optional[List[str]]:
    if isinstance(choices, list):
        lines: List[str] = []
        for i, c in enumerate(choices):
            if isinstance(c, str):
                chunk = c.strip()
            else:
                chunk = str(c).strip() if c is not None else ""
            if not chunk:
                return None
            lines.append(f"{chr(ord('A') + i)}. {chunk}")
        return lines if lines else None
    if isinstance(choices, dict):
        keys = sorted(choices.keys(), key=lambda k: str(k))
        lines = []
        for k in keys:
            v = choices[k]
            label = str(k).strip().upper()
            if len(label) != 1 or label < "A" or label > "Z":
                return None
            chunk = v.strip() if isinstance(v, str) else str(v).strip()
            if not chunk:
                return None
            lines.append(f"{label}. {chunk}")
        return lines if lines else None
    return None


def _record_to_mmlu_canonical_text(record: dict) -> Optional[str]:
    """Match mmlu_*_canonical.jsonl: prompt + response, or build the same shape from question/choices/answer."""
    prompt = record.get("prompt")
    response = record.get("response")
    if isinstance(prompt, str) and isinstance(response, str):
        return (prompt + response).strip() or None

    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    choices_obj = record.get("choices")
    if choices_obj is None:
        return None
    choice_lines = _choices_as_lines(choices_obj)
    if not choice_lines:
        return None
    n = len(choice_lines)
    letter = _answer_to_choice_letter(record.get("answer"), n)
    if letter is None:
        return None

    subject = record.get("subject")
    if isinstance(subject, str) and subject.strip():
        subj = subject.strip()
    else:
        subj = "general knowledge"

    intro = f"The following are multiple choice questions (with answers) about {subj}:\n\n"
    body = question.strip() + "\n" + "\n".join(choice_lines) + "\nAnswer:"
    full_prompt = intro + body
    full_response = " " + letter
    return (full_prompt + full_response).strip() or None


def _flatten_to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_flatten_to_text(v) for v in value if _flatten_to_text(v))
    if isinstance(value, dict):
        ordered_keys = (
            "question",
            "choices",
            "answer",
            "subject",
            "instruction",
            "input",
            "output",
            "text",
            "content",
            "body",
            "raw_content",
            "document",
        )
        parts: List[str] = []
        for key in ordered_keys:
            if key in value:
                chunk = _flatten_to_text(value[key]).strip()
                if chunk:
                    parts.append(f"{key}: {chunk}")
        for key, val in value.items():
            if key in ordered_keys:
                continue
            chunk = _flatten_to_text(val).strip()
            if chunk:
                parts.append(f"{key}: {chunk}")
        return " | ".join(parts)
    return ""


def _extract_text_from_record(record: dict, preferred_field: str) -> Optional[str]:
    canonical = _record_to_mmlu_canonical_text(record)
    if canonical:
        return canonical

    direct_text = _extract_text_from_json(record, preferred_field)
    if direct_text:
        return direct_text

    # Fallback for structured datasets (e.g., non-MMLU parquet).
    synthesized = _flatten_to_text(record).strip()
    return synthesized or None


def _normalize_text(text: str, min_chars: int) -> Optional[str]:
    text = " ".join(text.split())
    if len(text) < min_chars:
        return None
    return text


def _iter_jsonl_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line


def _iter_jsonl_zst_lines(path: Path) -> Iterable[str]:
    try:
        import zstandard as zstd
    except ImportError as e:
        raise RuntimeError(
            "Reading .jsonl.zst requires the 'zstandard' package. Install with: pip install zstandard"
        ) from e

    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = reader.read().decode("utf-8", errors="ignore").splitlines()
            for line in text_stream:
                yield line


def _load_from_jsonl_like(path: Path, preferred_field: str, min_chars: int, cap: int, texts: List[str]) -> None:
    line_iter = _iter_jsonl_zst_lines(path) if path.name.endswith(".jsonl.zst") else _iter_jsonl_lines(path)
    for line in line_iter:
        line = line.strip()
        if not line:
            continue
        text: Optional[str] = None
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    text = _extract_text_from_record(obj, preferred_field)
            except json.JSONDecodeError:
                text = line
        else:
            text = line

        if text is None:
            continue
        normalized = _normalize_text(text, min_chars)
        if normalized is None:
            continue

        texts.append(normalized)
        if cap > 0 and len(texts) >= cap:
            return


def _load_from_parquet(path: Path, preferred_field: str, min_chars: int, cap: int, texts: List[str]) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "Reading .parquet requires pyarrow. Install with: pip install pyarrow"
        ) from e

    table = pq.read_table(path)
    for row in table.to_pylist():
        if not isinstance(row, dict):
            continue
        value = _extract_text_from_record(row, preferred_field)
        if not value:
            continue
        normalized = _normalize_text(value, min_chars)
        if normalized is None:
            continue
        texts.append(normalized)
        if cap > 0 and len(texts) >= cap:
            return


def _discover_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if path.is_file():
        return [path]

    files: List[Path] = []
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith((".jsonl.zst", ".jsonl", ".txt", ".parquet")):
            files.append(p)
    files.sort()
    return files


def load_texts(path: str, preferred_field: str, min_chars: int, cap: int) -> List[str]:
    texts: List[str] = []
    files = _discover_files(path)
    if not files:
        raise ValueError(f"No supported files found under: {path}")

    for file_path in files:
        suffix_name = file_path.name
        if suffix_name.endswith(".parquet"):
            _load_from_parquet(file_path, preferred_field, min_chars, cap, texts)
        elif suffix_name.endswith(".jsonl.zst") or suffix_name.endswith(".jsonl") or suffix_name.endswith(".txt"):
            _load_from_jsonl_like(file_path, preferred_field, min_chars, cap, texts)
        if cap > 0 and len(texts) >= cap:
            break
    return texts


def to_labeled_lines(texts: Iterable[str], label: str) -> List[str]:
    return [f"{label} {t}" for t in texts]


def split_train_valid(lines: List[str], valid_frac: float, rng: random.Random) -> tuple[List[str], List[str]]:
    if not (0 <= valid_frac < 1):
        raise ValueError("--valid_frac must be in [0, 1).")
    shuffled = list(lines)
    rng.shuffle(shuffled)
    if valid_frac == 0:
        return shuffled, []
    n_valid = max(1, int(len(shuffled) * valid_frac))
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    return train, valid


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def build_dataset(
    hq_texts: List[str],
    lq_texts: List[str],
    dataset_name: str,
    output_dir: str,
    valid_frac: float,
    rng: random.Random,
) -> None:
    n = min(len(hq_texts), len(lq_texts))
    if n == 0:
        raise ValueError(f"Dataset {dataset_name} has no usable examples after filtering.")

    hq_sample = list(hq_texts)
    lq_sample = list(lq_texts)
    rng.shuffle(hq_sample)
    rng.shuffle(lq_sample)
    hq_sample = hq_sample[:n]
    lq_sample = lq_sample[:n]

    all_lines = to_labeled_lines(hq_sample, HQ_LABEL) + to_labeled_lines(lq_sample, LQ_LABEL)
    train_lines, valid_lines = split_train_valid(all_lines, valid_frac, rng)

    base = os.path.join(output_dir, dataset_name)
    train_path = f"{base}.train.txt"
    valid_path = f"{base}.valid.txt"
    write_lines(train_path, train_lines)
    write_lines(valid_path, valid_lines)

    print(f"[{dataset_name}] balanced per class: {n}")
    print(f"[{dataset_name}] train rows: {len(train_lines)} -> {train_path}")
    print(f"[{dataset_name}] valid rows: {len(valid_lines)} -> {valid_path}")


def main() -> None:
    args = get_args()
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    mmlu_texts = load_texts(args.mmlu_input, args.text_field, args.min_chars, args.max_per_source)
    refinedweb_texts = load_texts(args.refinedweb_input, args.text_field, args.min_chars, args.max_per_source)
    dclm_texts = load_texts(args.dclm_input, args.text_field, args.min_chars, args.max_per_source)

    print(f"Loaded MMLU rows: {len(mmlu_texts)}")
    print(f"Loaded RefinedWeb rows: {len(refinedweb_texts)}")
    print(f"Loaded DCLM rows: {len(dclm_texts)}")

    build_dataset(
        hq_texts=mmlu_texts,
        lq_texts=refinedweb_texts,
        dataset_name="mmlu_vs_refinedweb",
        output_dir=args.output_dir,
        valid_frac=args.valid_frac,
        rng=rng,
    )
    build_dataset(
        hq_texts=mmlu_texts,
        lq_texts=dclm_texts,
        dataset_name="mmlu_vs_dclm",
        output_dir=args.output_dir,
        valid_frac=args.valid_frac,
        rng=rng,
    )


if __name__ == "__main__":
    main()
