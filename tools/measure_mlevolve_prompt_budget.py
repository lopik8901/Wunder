"""Read-only prompt inventory. Prints sizes, never prompt contents or scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def measure_prompt(prompt: object, tokenizer=None) -> dict[str, int]:
    if isinstance(prompt, dict):
        parts = [str(prompt.get(key, "")) for key in ("system", "user", "assistant")]
    elif isinstance(prompt, str):
        parts = [prompt]
    else:
        raise ValueError("unexpected prompt format")
    result = {"characters": sum(len(p) for p in parts),
              "utf8_bytes": sum(len(p.encode("utf-8")) for p in parts)}
    if tokenizer is not None:
        result["content_tokens"] = sum(len(tokenizer.encode(p).ids) for p in parts)
    return result


def inventory(journal: Path, tokenizer=None) -> dict[str, object]:
    data = json.loads(journal.read_text(encoding="utf-8"))
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("journal nodes missing")
    rows = [measure_prompt(node["prompt_input"], tokenizer)
            for node in nodes if isinstance(node, dict) and isinstance(node.get("prompt_input"), (str, dict))]
    return {"prompt_count": len(rows), "prompts": rows,
            "max_characters": max((r["characters"] for r in rows), default=0),
            "max_utf8_bytes": max((r["utf8_bytes"] for r in rows), default=0),
            "tokenizer": "local_tokenizer_json" if tokenizer else "unavailable; no exact token claim",
            "chat_wrapper_tokens_included": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("journal", type=Path)
    parser.add_argument("--tokenizer-json", type=Path,
                        help="Optional local tokenizer.json; no downloads are performed")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer_json:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    print(json.dumps(inventory(args.journal, tokenizer), indent=2))


if __name__ == "__main__":
    main()
