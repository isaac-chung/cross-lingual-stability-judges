"""Core parsing logic for combining and validating conversation JSONL files."""

import json
import re
from collections import defaultdict
from pathlib import Path

# Pattern to extract language from filename: convo_{model}_{lang}_{datetime}.jsonl
FILENAME_PATTERN = re.compile(r"^convo_.+_([a-z]{2}(?:-[a-z]{2})?)_\d{8}-\d{6}\.jsonl$")


def extract_language_from_filename(file_path: Path) -> str | None:
    """Extract language code from filename if it matches the expected pattern.

    Expected format: convo_{model}_{lang}_{datetime}.jsonl
    Examples: convo_gpt-4.1-mini_et_20260127-143052.jsonl -> "et"
              convo_claude-3_en-us_20260127-143052.jsonl -> "en-us"

    Returns:
        Language code or None if pattern doesn't match.
    """
    match = FILENAME_PATTERN.match(file_path.name)
    if match:
        return match.group(1)
    return None


def parse_jsonl(file_path: Path) -> list[dict]:
    """Load and parse a single JSONL file, adding source_file field.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of conversation dicts with source_file added.
    """
    conversations = []
    source = str(file_path)

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data["source_file"] = source
                conversations.append(data)
            except json.JSONDecodeError as e:
                # Log warning but continue processing
                print(f"Warning: {file_path}:{line_num}: {e}")

    return conversations


def combine_files(paths: list[Path]) -> list[dict]:
    """Combine multiple JSONL files into a single list.

    Args:
        paths: List of paths to JSONL files.

    Returns:
        Combined list of all conversations.
    """
    all_conversations = []
    for path in paths:
        conversations = parse_jsonl(path)
        all_conversations.extend(conversations)
    return all_conversations


def get_language(conv: dict) -> str:
    """Get language from conversation, trying filename first, then metadata.

    Args:
        conv: Conversation dict with source_file and optionally _metadata.

    Returns:
        Language code or "unknown".
    """
    # Try filename first
    source_file = conv.get("source_file")
    if source_file:
        lang = extract_language_from_filename(Path(source_file))
        if lang and lang != "mixed":
            return lang

    # Fall back to metadata
    metadata = conv.get("_metadata", {})
    return metadata.get("language", "unknown")


def compute_stats(conversations: list[dict]) -> dict:
    """Compute summary statistics grouped by model and language.

    Args:
        conversations: List of conversation dicts.

    Returns:
        Dict with counts by model and by language.
    """
    by_model: dict[str, int] = defaultdict(int)
    by_language: dict[str, int] = defaultdict(int)
    by_model_language: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for conv in conversations:
        model = conv.get("model", "unknown")
        language = get_language(conv)

        by_model[model] += 1
        by_language[language] += 1
        by_model_language[model][language] += 1

    return {
        "total": len(conversations),
        "by_model": dict(by_model),
        "by_language": dict(by_language),
        "by_model_language": {m: dict(langs) for m, langs in by_model_language.items()},
    }


def group_by_model(conversations: list[dict]) -> dict[str, list[dict]]:
    """Group conversations by model.

    Args:
        conversations: List of conversation dicts.

    Returns:
        Dict mapping model names to lists of conversations.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for conv in conversations:
        model = conv.get("model", "unknown")
        grouped[model].append(conv)
    return dict(grouped)
