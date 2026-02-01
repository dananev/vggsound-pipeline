"""Label-based pre-filtering for VGGSound samples.

VGGSound has ~310 unique labels. Many clearly indicate music or speech
and can be filtered without ML inference, improving efficiency.

Categories:
- MUSIC: "playing *" instruments, singing, orchestra, etc.
- SPEECH: speaking, talking, whispering, etc.
- SFX: Everything else (potential sound effects)
"""

import re
from pathlib import Path

import orjson

from .extraction import VideoMetadata, parse_vggsound_csv

# Sports and non-music "playing" activities to exclude from music detection
NON_MUSIC_PLAYING = {
    "playing badminton",
    "playing basketball",
    "playing darts",
    "playing hockey",
    "playing lacrosse",
    "playing squash",
    "playing tennis",
    "playing volleyball",
    "playing ping pong",
    "playing table tennis",
    "playing pool",
    "playing billiards",
    "playing video games",
    "playing cards",
}

# Patterns that strongly indicate MUSIC
MUSIC_PATTERNS = [
    r"singing",
    r"orchestra",
    r"beat\s*box",
    r"humming",
    r"rapping",
    r"yodeling",
    r"whistling\s+tune",  # Whistling a tune vs whistling as SFX
    r"vocalizing",
    r"choir",
    r"chanting",
    r"opera",
    r"acappella",
    r"music",
]

# Patterns that strongly indicate SPEECH
SPEECH_PATTERNS = [
    r"speech",
    r"speaking",
    r"^talking",
    r"conversation",
    r"whispering",
    r"narrat",
    r"announcing",
    r"commentat",
    r"lecturing",
    r"preaching",
    r"praying",
    r"reading\s+(aloud|book)",
    r"reciting",
]

# Compile patterns for efficiency
MUSIC_REGEX = re.compile("|".join(MUSIC_PATTERNS), re.IGNORECASE)
SPEECH_REGEX = re.compile("|".join(SPEECH_PATTERNS), re.IGNORECASE)


def categorize_label(label: str) -> str:
    """Categorize a VGGSound label as music, speech, or sfx.

    Args:
        label: VGGSound label string

    Returns:
        Category string: "music", "speech", or "sfx"
    """
    label_lower = label.lower()

    # Check for non-music "playing" activities first
    if label_lower in NON_MUSIC_PLAYING:
        return "sfx"

    # "playing X" where X is an instrument is music
    if label_lower.startswith("playing "):
        return "music"

    if MUSIC_REGEX.search(label):
        return "music"
    if SPEECH_REGEX.search(label):
        return "speech"
    return "sfx"


def filter_by_labels(
    metadata_list: list[VideoMetadata],
) -> tuple[list[VideoMetadata], dict[str, list[str]]]:
    """Filter metadata by label categories.

    Args:
        metadata_list: List of VideoMetadata objects

    Returns:
        Tuple of:
        - List of SFX candidates (music/speech filtered out)
        - Dict with rejection reasons {"music": [...], "speech": [...]}
    """
    sfx_candidates = []
    rejected = {"music": [], "speech": []}

    for meta in metadata_list:
        category = categorize_label(meta.label)
        if category == "sfx":
            sfx_candidates.append(meta)
        else:
            rejected[category].append(meta.sample_id)

    return sfx_candidates, rejected


def get_unique_labels(csv_path: Path) -> set[str]:
    """Extract all unique labels from VGGSound CSV.

    Args:
        csv_path: Path to vggsound.csv

    Returns:
        Set of unique label strings
    """
    metadata = parse_vggsound_csv(csv_path)
    return {m.label for m in metadata}


def extract_and_categorize_labels(csv_path: Path, output_path: Path) -> dict:
    """Extract labels from CSV and categorize them.

    Creates a JSON file with categorized labels for review/editing.

    Args:
        csv_path: Path to vggsound.csv
        output_path: Path for output JSON

    Returns:
        Dict with categorized labels
    """
    labels = get_unique_labels(csv_path)

    categorized = {"music": [], "speech": [], "sfx": []}
    for label in sorted(labels):
        category = categorize_label(label)
        categorized[category].append(label)

    # Add summary stats
    result = {
        "total_labels": len(labels),
        "music_count": len(categorized["music"]),
        "speech_count": len(categorized["speech"]),
        "sfx_count": len(categorized["sfx"]),
        "categories": categorized,
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))

    print(f"Categorized {len(labels)} labels:")
    print(f"  Music: {len(categorized['music'])}")
    print(f"  Speech: {len(categorized['speech'])}")
    print(f"  SFX candidates: {len(categorized['sfx'])}")
    print(f"Saved to {output_path}")

    return result


def load_custom_categories(json_path: Path) -> dict[str, str]:
    """Load custom label categorizations from JSON.

    Allows manual override of automatic categorizations.

    Args:
        json_path: Path to JSON file with format:
            {"label": "category", ...}

    Returns:
        Dict mapping labels to categories
    """
    if not json_path.exists():
        return {}

    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())

    # Handle both formats:
    # 1. Direct mapping: {"label": "category"}
    # 2. Nested: {"categories": {"music": [...], "speech": [...], "sfx": [...]}}
    if "categories" in data:
        mapping = {}
        for category, labels in data["categories"].items():
            for label in labels:
                mapping[label] = category
        return mapping

    return data
