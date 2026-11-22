from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


ARTICLES = ("a", "an", "the")


def _word_ranges(low: str, needle: str) -> List[Tuple[int, int]]:
    """Whole-word, case-insensitive char ranges of *needle* in the lowercased *low*.

    Whole-word means the match is not glued to an alphanumeric on either side, so
    'cat' does not match inside 'category' and 'a cow' does not match 'a cowboy'.
    """
    n, out = len(low), []
    for m in re.finditer(re.escape(needle), low):
        s, e = m.start(), m.end()
        if s > 0 and low[s - 1].isalnum():
            continue
        if e < n and low[e].isalnum():
            continue
        out.append((s, e))
    return out


def article_span(prompt: str, word: str) -> str:
    """Return *word* prefixed by the article **the prompt itself uses**, else bare.

    Span-mode steering targets "article + noun" instead of the bare noun, and the
    span is then located by literal whole-word search. Hard-coding "a " breaks the
    moment the prompt says "an oven" (vowel-initial noun) or uses no article at all
    ("broccoli", "scissors"): the span is never found, the noun gets an empty token
    list, and it is steered on a degenerate span — or, when *every* word of a prompt
    fails, the loss detaches from the graph and autograd raises
    "element 0 of tensors does not require grad".

    So: try each article as the prompt would write it and fall back to the bare noun.
    """
    low, wl = prompt.lower(), word.lower().strip()
    for art in ARTICLES:
        if _word_ranges(low, f"{art} {wl}"):
            return f"{art} {word.strip()}"
    return word.strip()


def build_token_index_map(
    prompt: str,
    target_words: List[str],
    offset_mapping: Sequence[Tuple[int, int]],
) -> Dict[str, List[int]]:
    """Map each target word to the exact token indices that cover it.

    Uses the tokenizer's character ``offset_mapping`` (from a fast tokenizer's
    ``return_offsets_mapping=True``): we locate each target word's character span
    in *prompt* (whole-word, case-insensitive) and collect every token whose char
    span overlaps it.  This is exact — unlike substring matching against
    reconstructed token strings, it never attributes article tokens ("a"/"an") to
    a word merely because the letter appears in it, and it is unaffected by
    punctuation glued onto a token (e.g. "hydrant,").

    When one target word is contained in another ("dog" and "hot dog" in the same
    prompt), longest match wins: the occurrence of "dog" that sits inside "hot dog"
    belongs to "hot dog" alone, so the short word only keeps its own standalone
    occurrences.  Without this the two objects share tokens and their gradients mix.

    Returns a dict of word → sorted list of token indices (empty list if no match).
    """
    result: Dict[str, List[int]] = {w: [] for w in target_words}
    low = prompt.lower()

    claimed: List[Tuple[int, int]] = []
    # longest first so a containing word claims its range before the contained one looks
    for word in sorted(target_words, key=lambda w: -len(w.strip())):
        wl = word.lower().strip()
        if not wl:
            continue
        for s, e in _word_ranges(low, wl):
            if any(cs <= s and e <= ce for cs, ce in claimed):
                continue  # this occurrence belongs to a longer target word
            claimed.append((s, e))
            for ti, (a, b) in enumerate(offset_mapping):
                if b > a and a < e and b > s:  # token char-span overlaps [s, e)
                    result[word].append(ti)

    return {k: sorted(set(v)) for k, v in result.items()}
