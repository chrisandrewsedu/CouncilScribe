"""Layer 3: LLM-assisted speaker identification via llama-cpp-python."""

from __future__ import annotations

import json
import re
from typing import Optional

from . import config
from .models import Segment, SpeakerMapping


def load_llm(n_gpu_layers: int = -1):
    """Download and load a small GGUF model for speaker identification.

    Uses llama-cpp-python with GPU offloading when available.
    n_gpu_layers=-1 means offload all layers to GPU.
    """
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = hf_hub_download(
        repo_id=config.LLM_REPO,
        filename=config.LLM_FILENAME,
    )

    llm = Llama(
        model_path=model_path,
        n_ctx=config.LLM_CONTEXT_TOKENS,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    return llm


def unload_llm(llm) -> None:
    """Explicitly free LLM memory."""
    import gc

    import torch

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prompt_for_speaker_id(
    llm,
    segments: list[Segment],
    current_mappings: dict[str, SpeakerMapping],
    unknown_label: str,
    window_size: int = 30,
) -> Optional[SpeakerMapping]:
    """Ask the LLM to identify a single unknown speaker from context.

    Sends a ~2000-token transcript window around the unknown speaker's
    segments, with known identities annotated.
    """
    # Find segments from the unknown speaker
    unknown_indices = [
        i for i, s in enumerate(segments) if s.speaker_label == unknown_label
    ]
    if not unknown_indices:
        return None

    # Build context window around the first occurrence
    center = unknown_indices[0]
    start = max(0, center - window_size // 2)
    end = min(len(segments), center + window_size // 2)
    window = segments[start:end]

    # Format transcript excerpt
    lines = []
    for seg in window:
        mapping = current_mappings.get(seg.speaker_label)
        if mapping and mapping.speaker_name:
            speaker = mapping.speaker_name
        elif seg.speaker_label == unknown_label:
            speaker = f"[UNKNOWN - {unknown_label}]"
        else:
            speaker = seg.speaker_label
        lines.append(f"{speaker}: {seg.text}")

    transcript_excerpt = "\n".join(lines)

    known_speakers = []
    for label, m in current_mappings.items():
        if m.speaker_name:
            known_speakers.append(f"  - {label} = {m.speaker_name}")
    known_section = "\n".join(known_speakers) if known_speakers else "  (none identified yet)"

    prompt = f"""You are analyzing a city council meeting transcript to identify speakers.

Known speakers:
{known_section}

Unknown speaker to identify: {unknown_label}

Transcript excerpt:
---
{transcript_excerpt}
---

Based on the context, who is {unknown_label}? Consider:
- How other speakers address them
- What topics they discuss and their role
- Conversational patterns and turn-taking

Respond with ONLY a JSON object:
{{"name": "Speaker Name or null", "reasoning": "brief explanation"}}"""

    response = llm(
        prompt,
        max_tokens=150,
        temperature=0.1,
        stop=["\n\n"],
    )

    text = response["choices"][0]["text"].strip()
    return _parse_llm_response(text, unknown_label)


def _parse_llm_response(
    text: str, speaker_label: str
) -> Optional[SpeakerMapping]:
    """Parse LLM JSON response into a SpeakerMapping."""
    # Try to extract JSON from the response
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    name = data.get("name")
    if not name or name.lower() == "null" or name.lower() == "unknown":
        return None

    return SpeakerMapping(
        speaker_label=speaker_label,
        speaker_name=name,
        confidence=0.75,
        id_method="llm",
    )


def llm_identify_speakers(
    llm,
    segments: list[Segment],
    current_mappings: dict[str, SpeakerMapping],
) -> dict[str, SpeakerMapping]:
    """Identify all unresolved speakers using the LLM.

    This function is designed to be passed as llm_identify_fn to
    identify.identify_speakers().
    """
    all_labels = {seg.speaker_label for seg in segments}
    unresolved = [
        label for label in all_labels
        if label not in current_mappings or not current_mappings[label].speaker_name
    ]

    results: dict[str, SpeakerMapping] = {}
    for label in unresolved:
        mapping = prompt_for_speaker_id(llm, segments, current_mappings, label)
        if mapping:
            results[label] = mapping

    return results
