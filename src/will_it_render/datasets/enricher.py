from __future__ import annotations

from typing import List

from will_it_render.datasets.csv_reader import ObjectEntry

_SYSTEM_PROMPT = (
    "You are a creative scene description generator. "
    "Given an object or sound, write vivid, realistic short scene descriptions (1-2 sentences) "
    "that feature the characteristic sounds the object makes. "
    "Each description should be a unique scenario."
)


def build_enricher(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", device: str = "cuda"):
    from transformers import pipeline
    return pipeline(
        "text-generation",
        model=model_name,
        device=device,
        torch_dtype="auto",
    )


def enrich_entry(entry: ObjectEntry, pipe, n: int = 10) -> List[str]:
    """Generate n varied prompts for an ObjectEntry using the LLM pipeline."""
    user_msg = (
        f"Generate {n} different scene descriptions featuring a {entry.object_name}. "
        f"Focus on scenes where these sounds are prominent: {', '.join(entry.target_words)}. "
        f"Output one description per line, no numbering."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    result = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.9)
    raw = result[0]["generated_text"]
    if isinstance(raw, list):
        # chat format: last message is assistant reply
        raw = raw[-1]["content"]
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    return lines[:n]
