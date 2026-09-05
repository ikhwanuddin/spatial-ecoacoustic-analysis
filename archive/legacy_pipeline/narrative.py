#!/usr/bin/env python3
"""LLM narrative card for the bacpipe embedding reports.

Builds a small numeric digest from the report stats, asks an LLM to turn it into
grounded prose, and falls back to a rule-based summary when no LLM is reachable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

# Any OpenAI-compatible chat endpoint.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "minimax/minimax-m3:free"
KEY_ENV_VARS = ("OPENROUTER_API_KEY", "NARRATIVE_API_KEY", "XAI_API_KEY")
KEY_FILE = "~/.openrouter_api_key"

SYSTEM_PROMPT = """You write the interpretation card for an ecoacoustics evaluation report.

The report compares four audio rendering methods on the same recordings:
- mono: single microphone channel (the baseline)
- sa: signal averaging across channels
- bf_LabIR: beamforming steered with laboratory-measured impulse responses
- bf_SPIR: beamforming steered with in-situ measured impulse responses

Each matched window scores every method by its cosine distance from a set of
habitat noise-reference embeddings. A larger noise distance means the embedding
sits further from background noise, which is the intended effect of beamforming.
Deltas are reported against mono, with a one-sided Wilcoxon signed-rank test and
Cliff's delta as effect size.

Rules:
- Use ONLY the numbers in the digest. Never invent values, species, or citations.
- Quote the numbers you rely on, and always pair a p-value with its effect size.
- Note when an effect is statistically significant but small in absolute terms.
- Say plainly when a method fails to beat the baseline.
- Be specific and neutral. No hype, no recommendations about future work.
- British-neutral scientific English, third person, no first person.

Return STRICT JSON only, no markdown fence:
{"headline": "<= 110 characters",
 "findings": ["3 to 5 sentences, each a complete standalone sentence"],
 "caveats": ["1 to 3 sentences naming the real limits of this evidence"]}"""


def _round(value: Any, digits: int = 5) -> Any:
    """Trim float noise so the model never quotes 17 significant digits."""
    if isinstance(value, float):
        if value != 0 and abs(value) < 10 ** -digits:
            return float(f"{value:.2e}")
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


def build_digest(
    *,
    model: str,
    date_str: str,
    location: str,
    method_names: Sequence[str],
    stats: Dict[str, Any],
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Reduce the full stats blob to the few numbers worth narrating."""
    overview = stats.get("overview", {})
    matched = (stats.get("matched_analysis") or {}).get("summary_stats", {})
    noise = stats.get("noise_analysis") or {}
    shared = (stats.get("shared_cluster_analysis") or {}).get("summary", {})

    per_method = {
        row["method"]: {
            "n_embeddings": row.get("n_embeddings"),
            "n_unique_clusters": row.get("n_unique_clusters"),
            "noise_pct": row.get("noise_pct"),
            "intra_method_cosine": row.get("cosine_sim"),
        }
        for row in stats.get("per_method", [])
    }
    for row in noise.get("per_method", []) if noise.get("available") else []:
        per_method.setdefault(row["method"], {})
        per_method[row["method"]]["mean_noise_distance"] = row.get("mean_noise_distance")
        per_method[row["method"]]["delta_vs_mono"] = row.get("delta_vs_mono")

    return _round({
        "model": model,
        "date": date_str,
        "location": location,
        "embedding_dim": overview.get("embedding_dim"),
        "n_points": overview.get("total_embeddings"),
        "n_matched_windows": matched.get("n_matched_windows"),
        "methods": list(method_names),
        "per_method": per_method,
        "hypothesis_tests": matched.get("hypothesis_tests", {}),
        "beam_distribution": matched.get("beam_distribution", {}),
        "clustering": {
            "n_clusters": overview.get("n_clusters"),
            "noise_pct": overview.get("noise_pct"),
            "min_cluster_size": parameters.get("min_cluster_size"),
            "tiers": shared,
        },
    })


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "N/A"


def deterministic_narrative(digest: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based summary used when no LLM is available."""
    tests = digest.get("hypothesis_tests", {})
    findings: List[str] = []
    winners, losers = [], []
    for method, t in tests.items():
        delta = t.get("mean_delta_vs_mono")
        sign = "+" if isinstance(delta, (int, float)) and delta >= 0 else ""
        sentence = (
            f"{method} shifts the mean noise distance by {sign}{_fmt(delta)} against mono "
            f"({t.get('win_rate_pct')}% of {digest.get('n_matched_windows')} matched windows, "
            f"Wilcoxon p={_fmt(t.get('wilcoxon_p_value'), 5)}, Cliff's delta={t.get('cliffs_delta')})."
        )
        findings.append(sentence)
        if t.get("is_significant_p05") and isinstance(delta, (int, float)) and delta > 0:
            winners.append(method)
        elif isinstance(delta, (int, float)) and delta <= 0:
            losers.append(method)

    clustering = digest.get("clustering", {})
    findings.append(
        f"HDBSCAN on the {digest.get('embedding_dim')}-dimensional {digest.get('model')} space "
        f"returns {clustering.get('n_clusters')} clusters over {digest.get('n_points')} points "
        f"with {clustering.get('noise_pct')}% unclustered."
    )
    if winners:
        headline = f"{', '.join(winners)} beats mono; {', '.join(losers) or 'no method'} does not"
    else:
        headline = "No method separates from the mono baseline in this run"
    return {
        "headline": headline,
        "findings": findings,
        "caveats": [
            "Generated without an LLM: this is a direct restatement of the table above, not an interpretation.",
            "Noise distance is a cosine distance in embedding space and is not a dB signal-to-noise ratio.",
        ],
        "source": "deterministic",
    }


def _api_key() -> Optional[str]:
    """Key from the environment first, then ~/.xai_api_key."""
    for name in KEY_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    path = os.path.expanduser(KEY_FILE)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip() or None
    return None


def _call_llm(digest: Dict[str, Any], llm_model: str, base_url: str, timeout: int) -> Dict[str, Any]:
    key = _api_key()
    if not key:
        raise RuntimeError(f"no API key: set {KEY_ENV_VARS[0]} or write it to {KEY_FILE}")
    payload = {
        "model": llm_model,
        "temperature": 0,
        "max_tokens": 3000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(digest, ensure_ascii=False, indent=1)},
        ],
    }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    choice = body["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("LLM response was cut off by the token limit")
    text = choice["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(text)
    if not parsed.get("headline") or not parsed.get("findings"):
        raise RuntimeError("LLM returned an incomplete narrative")
    parsed["source"] = llm_model
    return parsed


def generate_narrative(
    digest: Dict[str, Any],
    *,
    mode: str = "auto",
    llm_model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Return a narrative dict; never raises."""
    if mode == "off":
        return deterministic_narrative(digest)
    last_error = None
    for attempt in (1, 2, 3):
        try:
            narrative = _call_llm(digest, llm_model, base_url, timeout)
            break
        except Exception as exc:  # noqa: BLE001 - a broken narrative must not kill the report
            last_error = exc
            print(f"  narrative: attempt {attempt} failed ({type(exc).__name__}: {exc})")
            # free-tier endpoints return sporadic 402/429 under load
            time.sleep(5 * attempt)
    else:
        print(f"  narrative: falling back to the deterministic summary ({last_error})")
        return deterministic_narrative(digest)
    narrative["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return narrative


def load_or_generate_narrative(
    output_dir,
    model: str,
    digest: Dict[str, Any],
    *,
    mode: str = "auto",
    llm_model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> Dict[str, Any]:
    """Reuse a cached narrative when the numbers have not changed."""
    from pathlib import Path

    cache_path = Path(output_dir) / f"narrative_{model}.json"
    fingerprint = hashlib.sha256(
        json.dumps(digest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]

    if mode == "auto" and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("digest_fingerprint") == fingerprint:
                print(f"  narrative: reusing cached card ({cached.get('source')})")
                return cached
        except (OSError, ValueError):
            pass

    print(f"  narrative: writing interpretation card via {llm_model if mode != 'off' else 'rule-based template'}...")
    narrative = generate_narrative(digest, mode=mode, llm_model=llm_model, base_url=base_url)
    narrative["digest_fingerprint"] = fingerprint
    try:
        cache_path.write_text(json.dumps(narrative, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return narrative
