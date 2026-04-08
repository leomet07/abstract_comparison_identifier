"""
Comparison detector combining:
1. Lexical/POS-based detection (fast, catches obvious comparatives)
2. NLI zero-shot detection (catches implicit/subtle comparisons)
"""

import spacy
from transformers import pipeline
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# --- Approach 1: Lexical + POS ---

COMPARISON_MARKERS = {
    "compared to",
    "in comparison",
    "rather than",
    "as opposed to",
    "unlike",
    "whereas",
    "while",
    "instead of",
    "on the other hand",
    "more than",
    "less than",
    "better than",
    "worse than",
}

COMPARATIVE_POS = {"JJR", "RBR"}  # comparative adj/adv


def check_lexical(text):
    """Check for surface-level comparison cues."""
    lower = text.lower()

    # keyword scan
    markers_found = [m for m in COMPARISON_MARKERS if m in lower]

    # POS tag scan for comparatives/superlatives
    doc = nlp(text)
    comp_tokens = [tok.text for tok in doc if tok.tag_ in COMPARATIVE_POS]

    # dependency: look for "than" governed by a comparative
    than_comparisons = []
    for tok in doc:
        if tok.text.lower() == "than" and tok.head.tag_ in COMPARATIVE_POS:
            than_comparisons.append(f"{tok.head.text} than ...")

    score = min(1.0, (len(markers_found) + len(comp_tokens)) * 0.4)
    return {
        "score": round(score, 2),
        "markers": markers_found,
        "comparative_tokens": comp_tokens,
        "than_phrases": than_comparisons,
    }


# --- Approach 2: NLI zero-shot ---


def check_nli(text):
    """Use NLI to detect implicit comparisons."""
    result = nli(
        text,
        candidate_labels=["comparison between things", "no comparison"],
        hypothesis_template="This text contains {}.",
    )
    comp_idx = result["labels"].index("comparison between things")
    score = result["scores"][comp_idx]
    return {"score": round(score, 3)}


# --- Combined ---


def detect_comparison(text, nli_weight=0.6, lexical_weight=0.4):
    """Combine both signals into a single verdict."""
    lex = check_lexical(text)
    if lex["score"] >= 0.8:  # if lexical is strong enough, then just short circuit
        return {
            "text": text,
            "lexical": lex,
            "nli": None,
            "combined_score": lex["score"],
            "has_comparison": True,
        }

    nli_result = check_nli(text)
    combined = (nli_result["score"] * nli_weight) + (lex["score"] * lexical_weight)
    return {
        "text": text,
        "lexical": lex,
        "nli": nli_result,
        "combined_score": round(combined, 3),
        "has_comparison": combined > 0.5,
    }


def batch_nli(texts, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="NLI"):
        chunk = texts[i : i + batch_size]
        out = nli(
            chunk,
            candidate_labels=["comparison between things", "no comparison"],
            hypothesis_template="This text contains {}.",
            batch_size=batch_size,
        )
        if isinstance(out, dict):
            out = [out]
        results.extend(out)
    return results


def detect_comparisons_batch(
    texts, nli_weight=0.6, lexical_weight=0.4, lexical_threshold=0.8
):
    """Batch comparison detection — runs NLI only where needed."""
    print("About to run lexical results for all texts")
    lex_results = [check_lexical(t) for t in tqdm(texts)]
    print("Finished lexical results.")

    # split into "already decided" vs "needs NLI"
    needs_nli = []
    for i, lex in enumerate(lex_results):
        if lex["score"] < lexical_threshold:
            needs_nli.append(i)

    # batch NLI for the uncertain ones
    nli_scores = {}
    if needs_nli:
        nli_texts = [texts[i] for i in needs_nli]
        print(f"About to run NLI for {len(nli_texts)} texts")
        nli_results = batch_nli(nli_texts)
        print("Finished running nli")
        # nli() returns a single dict if given one string, list otherwise
        if isinstance(nli_results, dict):
            nli_results = [nli_results]
        for idx, res in zip(needs_nli, nli_results):
            comp_idx = res["labels"].index("comparison between things")
            nli_scores[idx] = res["scores"][comp_idx]

    # assemble results
    out = []
    for i, text in enumerate(texts):
        lex = lex_results[i]
        if i in nli_scores:
            ns = round(nli_scores[i], 3)
            combined = (ns * nli_weight) + (lex["score"] * lexical_weight)
            nli_out = {"score": ns}
        else:
            combined = lex["score"]
            nli_out = None

        out.append(
            {
                "text": text,
                "lexical": lex,
                "nli": nli_out,
                "combined_score": round(combined, 3),
                "has_comparison": combined > 0.5,
            }
        )
    return out


if __name__ == "__main__":
    examples = [
        # obvious
        "Python is faster than Java for prototyping.",
        # implicit
        "The first model struggles where the second excels.",
        # subtle / no explicit markers
        "Aluminum frames sacrifice comfort for weight savings.",
        # no comparison
        "The restaurant opens at 9am on weekdays.",
        # hedged comparison
        "Some argue that solar is becoming more viable than nuclear.",
    ]

    results = detect_comparisons_batch(examples)

    for sent, result in zip(examples, results):
        flag = "YES" if result["has_comparison"] else " NO"
        print(f"[{flag}] (score={result['combined_score']:.2f}) {sent}")
        if result["lexical"]["markers"] or result["lexical"]["comparative_tokens"]:
            print(f"       lexical cues: {result['lexical']}")
        print()
