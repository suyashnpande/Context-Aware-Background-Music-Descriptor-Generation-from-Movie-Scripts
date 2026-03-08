#!/usr/bin/env python3
"""
Multi-Genre Schema Extractor for Background Music Scoring
Using Gemini 2.5 Flash (Free Tier)
----------------------------------------------------------
Sends ENTIRE script in one API call — no sampling, no chunking.
Reads ALL .txt scripts from movie_scripts/ folder.
Saves unified_scoring_schema.json back into movie_scripts/.

Usage:
    python schema_extractor_gemini.py

No arguments needed.

Expected structure:
    Movie_extraction/
    ├── schema_extractor_gemini.py
    └── movie_scripts/
        ├── Avatar.txt
        ├── Saving-Private-Ryan.txt
        └── ...
"""

import json
import os
import re
import sys
import time
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL        = "gemini-2.5-flash"
SCRIPTS_DIR  = "movie_scripts"
OUTPUT_FILE  = os.path.join(SCRIPTS_DIR, "unified_scoring_schema.json")
SCENE_DELAY  = 8        # seconds between API calls (free tier: 10 RPM)

# Gemini 2.5 Flash context window = 1,000,000 tokens
# Average feature script = ~48,000 tokens = 4.8% of context
# Entire script fits easily in one call — no sampling needed

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

PER_SCRIPT_PROMPT = """
You are an expert film music supervisor and score composer.
You will receive a COMPLETE screenplay. Your job is to define the annotation schema
that will later be used to score background music for every scene in this film.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO PROCESS THIS SCRIPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read the script scene by scene, in order, from beginning to end.

For each scene, ask yourself:
  "What does a composer need to know about THIS scene to write the right music —
   given everything that has already happened in the story so far?"

As you move through scenes, maintain a running memory of:
  - What has happened to each character so far
  - What emotional state the audience is in
  - What narrative information has been revealed
  - What tensions, relationships, and conflicts have been established
  - Whether the current scene is a callback, escalation, or reversal of a past event

This context CHANGES the music. For example:
  - A quiet scene early in the film scores differently than the same quiet scene
    after a major trauma has occurred
  - A character walking alone scores differently before vs after we know they are being hunted
  - A reunion scores differently depending on whether it is joyful, complicated, or dangerous

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR OUTPUT GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After reading the entire script scene by scene with full narrative context, decide for yourself:
  - What aspects of a scene matter for how background music should sound
  - What those aspects are called (you name the fields)
  - What values those aspects can take across all scenes in this script
  - Which of those fields change meaning depending on what has happened earlier in the story

Do not follow any preset list of categories. Use your own judgment as a composer.
Every field you define must be something that would genuinely change a scoring decision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY a valid JSON object. No markdown. No explanation. No backticks.

{
  "genre": ["<genre1>", "<genre2>"],
  "total_scenes_found": <number>,
  "narrative_arc_observed": "<one line describing the emotional journey of this script>",
  "fields": {
    "<field_name>": {
      "type": "<categorical | numeric | boolean | array>",
      "scoring_impact": "<high | medium | low>",
      "context_dependent": <true if this field's value is influenced by prior scene history>,
      "why_needed": "<one line: what specific music decision this drives>",
      "values": <list for categorical/array | {"min":1,"max":10} for numeric | [true,false] for boolean>
    }
  }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Process scenes IN ORDER — never skip, never reorder
- Every field must be reusable across ALL scenes, not specific to one moment
  BAD:  "gunshot_trigger_moment"    <- one scene only
  GOOD: "emotional_shift_trigger"   <- applies to any scene with an emotional turn
- Values must be specific: "nature_jungle" not "nature", "border_interrogation" not "interrogation"
- Numeric fields: always 1-10 scale
- Categorical fields: FEWER than 30 values — generalize if needed
- Do NOT include scene_id or scene_text as fields
- Return ONLY the JSON
"""

MERGE_PROMPT = """
You are a film music supervisor.

Below are field schemas extracted from multiple complete movie scripts of different genres.
Merge them into ONE unified schema that works across ALL genres.

Rules:
- Keep ALL unique fields
- Merge values (union of all values per field)
- Mark which genres each field applies to, or "all" if universal
- Remove duplicate fields — if two fields mean the same thing, unify them
- Re-evaluate scoring_impact across all genres
- Categorical fields must have FEWER than 30 total values after merging — generalize if exceeded
- Remove any fields that are too scene-specific or appeared in only one script

Return ONLY a valid JSON object. No markdown. No explanation.

Format:
{
  "total_scripts_analyzed": <number>,
  "genres_covered": ["<genre1>", "<genre2>", ...],
  "fields": {
    "<field_name>": {
      "type": "<categorical | numeric | boolean | array>",
      "scoring_impact": "<high | medium | low>",
      "why_needed": "<what scoring decision this drives>",
      "relevant_genres": ["<genre>"] or ["all"],
      "values": <merged and capped values>
    }
  },
  "field_priority": {
    "essential": ["<fields needed for every genre>"],
    "genre_specific": {
      "<genre>": ["<fields only needed for this genre>"]
    }
  }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_scenes(text: str) -> int:
    """Count INT./EXT. scene headers in script."""
    return len(re.findall(
        r'^\s*(INT|EXT|INT/EXT|I/E)[\.\s]',
        text, re.MULTILINE | re.IGNORECASE
    ))


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def load_scripts() -> list[dict]:
    """Load all .txt files from movie_scripts/ folder."""
    if not os.path.isdir(SCRIPTS_DIR):
        print(f"ERROR: Folder '{SCRIPTS_DIR}/' not found.")
        print(f"  Create a '{SCRIPTS_DIR}/' folder with your .txt scripts")
        print(f"  and place this script next to it.")
        sys.exit(1)

    scripts = []
    for fname in sorted(os.listdir(SCRIPTS_DIR)):
        if fname.endswith(".txt"):
            fpath = os.path.join(SCRIPTS_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            scripts.append({
                "name":   fname.replace(".txt", ""),
                "file":   fname,
                "text":   text,
                "chars":  len(text),
                "tokens": estimate_tokens(text),
                "scenes": count_scenes(text)
            })

    return scripts


def parse_json(raw: str) -> dict:
    """
    Safely extract JSON from response.
    Handles cases where Gemini adds preamble text before the JSON object.
    """
    raw   = raw.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(raw[start:end])


def call_gemini(model, prompt: str, retries: int = 3) -> str:
    """Call Gemini with retry on rate limit errors."""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err:
                wait = 20 * (attempt + 1)
                print(f"  Rate limit hit. Waiting {wait}s... (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gemini call failed after {retries} retries")


def print_schema(schema: dict):
    """Print unified schema to terminal."""
    print("\n" + "=" * 65)
    print("  UNIFIED SCORING SCHEMA")
    print("=" * 65)
    print(f"  Scripts analyzed : {schema.get('total_scripts_analyzed', '?')}")
    print(f"  Genres covered   : {', '.join(schema.get('genres_covered', []))}")
    print("=" * 65)

    priority   = schema.get("field_priority", {})
    essential  = priority.get("essential", [])
    genre_spec = priority.get("genre_specific", {})
    fields     = schema.get("fields", {})

    print(f"\n[ESSENTIAL] {len(essential)} fields — needed for every genre\n")
    for f in essential:
        info = fields.get(f, {})
        print(f"  > {f}")
        print(f"    type   : {info.get('type','?')}   impact: {info.get('scoring_impact','?')}")
        print(f"    values : {str(info.get('values',''))[:80]}")
        print(f"    why    : {info.get('why_needed','')}\n")

    if genre_spec:
        print(f"\n[GENRE-SPECIFIC] fields\n")
        for genre, gfields in genre_spec.items():
            print(f"  [{genre.upper()}]")
            for f in gfields:
                info = fields.get(f, {})
                print(f"    > {f:35s} {str(info.get('values',''))[:50]}")
            print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run():
    # ── API Key ────────────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        print("  Get your free key -> https://aistudio.google.com/app/apikey")
        print("  Then: export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2          # deterministic, consistent field names
        }
    )

    # ── Load all scripts ───────────────────────────────────────────────────
    print(f"\nReading scripts from: {os.path.abspath(SCRIPTS_DIR)}/\n")
    scripts = load_scripts()

    if not scripts:
        print(f"ERROR: No .txt files found in {SCRIPTS_DIR}/")
        sys.exit(1)

    print(f"{'Script':<40} {'Chars':>10} {'~Tokens':>10} {'Scenes':>8}")
    print("-" * 72)
    for s in scripts:
        print(f"  {s['name']:<38} {s['chars']:>10,} {s['tokens']:>10,} {s['scenes']:>8}")

    total_tokens = sum(s["tokens"] for s in scripts)
    print("-" * 72)
    print(f"  {'TOTAL':<38} {sum(s['chars'] for s in scripts):>10,} {total_tokens:>10,}")
    print(f"\n  Gemini 2.5 Flash context window : 1,000,000 tokens")
    print(f"  Each script sent in ONE full API call — no chunking, no sampling")

    # ── Step 1: Full script → per-script schema ────────────────────────────
    print(f"\n{'-'*65}")
    print(f"  STEP 1: Extracting fields from each complete script")
    print(f"  Model       : {MODEL}")
    print(f"  Input       : entire script per call")
    print(f"  Temperature : 0.2")
    print(f"{'-'*65}")

    per_script_schemas = []

    for i, script in enumerate(scripts):
        print(f"\n  [{i+1}/{len(scripts)}] {script['name']}")
        print(f"    sending {script['tokens']:,} tokens (~{script['scenes']} scenes) to Gemini...")

        # Full script — no sampling at all
        prompt = (
            PER_SCRIPT_PROMPT
            + f"\n\nScript Title: {script['name']}\n\n"
            + "=" * 60 + "\n\n"
            + script["text"]
        )

        try:
            raw    = call_gemini(model, prompt)
            schema = parse_json(raw)
            genres         = schema.get("genre", ["unknown"])
            nfields        = len(schema.get("fields", {}))
            scenes_found   = schema.get("total_scenes_found", "?")
            print(f"    OK  genre={genres}  |  fields={nfields}  |  scenes seen={scenes_found}")
            per_script_schemas.append({"script": script["name"], **schema})
        except Exception as e:
            print(f"    FAILED: {e}")

        # Respect free tier: 10 RPM
        if i < len(scripts) - 1:
            print(f"    Waiting {SCENE_DELAY}s (free tier: 10 req/min)...")
            time.sleep(SCENE_DELAY)

    if not per_script_schemas:
        print("\nERROR: All scripts failed. Check your API key.")
        sys.exit(1)

    # ── Step 2: Merge all per-script schemas ──────────────────────────────
    print(f"\n{'-'*65}")
    print(f"  STEP 2: Merging {len(per_script_schemas)} schemas into unified schema...")
    print(f"{'-'*65}\n")

    if len(per_script_schemas) == 1:
        s = per_script_schemas[0]
        unified = {
            "total_scripts_analyzed": 1,
            "genres_covered": s.get("genre", ["unknown"]),
            "fields": s.get("fields", {}),
            "field_priority": {
                "essential": list(s.get("fields", {}).keys()),
                "genre_specific": {}
            }
        }
    else:
        merge_input = (
            MERGE_PROMPT
            + f"\n\nSchemas from {len(per_script_schemas)} scripts:\n\n"
            + json.dumps(per_script_schemas, indent=2)
        )
        raw     = call_gemini(model, merge_input)
        unified = parse_json(raw)

    unified["total_scripts_analyzed"] = len(per_script_schemas)

    # ── Save ───────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    print_schema(unified)
    print(f"\nDone.")
    print(f"  Schema saved to : {os.path.abspath(OUTPUT_FILE)}")
    print(f"  Total fields    : {len(unified.get('fields', {}))}")
    print(f"  Scripts used    : {len(per_script_schemas)}\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()