#!/usr/bin/env python3
"""
Scene-Level Music Scoring Annotator — Field-Aware Context
----------------------------------------------------------
- Field-aware state machine context (not scene log)
- Safe state merging (LLM cannot overwrite previous memory)
- Schema validation on every annotation
- Robust scene parser (handles header variations)
- scene_text included in output for model training
- Confidence score per annotation
- Hallucinated character detection

Place next to movie_scripts/ folder and run:
    python annotator.py

Output saved to annotations/ folder.
"""

import json
import math
import os
import re
import sys
import time
import google.generativeai as genai
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL              = "gemini-2.5-flash"
SCRIPTS_DIR        = "movie_scripts"
ANNOTATIONS_DIR    = "annotations"
OUTPUT_TOKEN_LIMIT = 7500
TOKENS_PER_SCENE   = 600
CALL_DELAY         = 10
CACHE_TTL_SECONDS  = 3600

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA — allowed values per field (used for validation)
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_SPEC = {
    "narrative_stage":                  ["Beginning_Setup","Middle_Development","Climax_Turning_Point","End_Resolution","Specific_Narrative_Device"],
    # EXPANDED: added Awe_Wonder (Avatar, sci-fi, nature films need this)
    "emotional_core":                   ["Positive_Uplifting","Negative_Distressing","Neutral_Complex","Tension_Action","Awe_Wonder"],
    "tension_level":                    (1, 10),
    "pacing_intensity":                 (1, 10),
    "scene_setting":                    ["Interior_Confined","Natural_Outdoor","Urban_Manmade","Military_Conflict_Zone","Abstract_Otherworldly"],
    "character_focus":                  ["Protagonist_Individual","Protagonist_Duo","Antagonist_Individual","Supporting_Character","Group_Collective","Neutral_Observer","Specific_Creature_Threat"],
    "emotional_dramatic_shift_trigger": [True, False],
    "emotional_intensity":              (1, 10),
    "action_intensity":                 (1, 10),
    # conflict_nature = the TYPE OF DANGER/PRESSURE in the scene (external or situational)
    "conflict_nature":                  ["Physical_Danger","Psychological_Tension","Interpersonal_Conflict","Moral_Dilemma","Environmental_Threat","Unknown_Threat","Time_Pressure","Claustrophobic",None],
    # conflict_type = the STRUCTURAL LEVEL of conflict (personal vs societal vs cosmic)
    "conflict_type":                    ["Internal_Personal","Interpersonal_Social","Large_Scale_Combat","Ideological_Political","Environmental_Survival","Existential_Abstract","None"],
    # EXPANDED: added Curious, Defiant, Terrified, Conflicted, Frustrated, Shocked,
    #           Exhausted, Hopeful, Excited, Grieving, Resigned, Protective, Amazed, Ruthless
    "character_internal_state":         [
        "Determined","Anxious","Confused","Angry","Sad","Calm","Focused","Vulnerable",
        "Despairing","Triumphant","Disoriented","Suspicious","Manipulative","Manipulated",
        "Self_Deceiving","Content",
        "Curious","Defiant","Terrified","Conflicted","Frustrated","Shocked",
        "Exhausted","Hopeful","Excited","Grieving","Resigned","Protective","Amazed","Ruthless",
        None
    ],
    "character_transformation":         [True, False],
    # NOTE: Internal_Monologue (not Internal_Monologue_Reflection) — fixed typo
    "reality_distortion_effect":        ["Present_Reality","Memory_Flashback","Dream_Fantasy","Internal_Monologue","Surreal_Psychedelic","Degraded_Fragmented","Bioluminescent_Ethereal","Stylized_Visuals"],
    "memory_state_degradation":         ["Clear_Intact","Fading_Degraded","Fragmented_Confused","Traumatic_Distorted","Constructed_False","Desire_To_Forget","Reliance_External_Aids",None],
    "musical_cue_type":                 ["Original_Score_Underscore","Leitmotif_Thematic","Source_Music_Diegetic","Silence_Dramatic","Soundscape_Natural_Tribal","Genre_Specific_Performance"],
    # EXPANDED: added Symbolic_Echo, Thematic_Callback
    "foreshadowing_callback":           ["None","Past_Trauma_Echo","Impending_Danger_Loss","Character_Arc_Development","Moral_Consequence","Prophecy_Legend","Symbolic_Echo","Thematic_Callback",None],
    # EXPANDED: added antagonistic/alliance/strained dynamics missing from original list
    "relationship_status":              [
        "Pre_Relationship","New_Potential","Developing_Attraction","Established_Intimate",
        "Strained_Conflict","Strained_Alliance","Breakup_Separation","Post_Breakup_Recovery",
        "Memory_Reflection","Developing_Mentorship","Antagonistic_Rivalry","Antagonistic",
        "Open_Hostility","Alliance_Under_Pressure","Reluctant_Alliance","Mentor_Student",
        None
    ],
    # REDESIGNED: now a free-text string — motif NAME is more useful than motif TYPE
    # Validator will accept any non-empty string instead of checking against a fixed list
    "symbolic_recurring_motif":         None,   # None = free-text field, validated differently
    "sense_of_repetition":              [True, False],
    "moral_ambiguity":                  (1, 10),
    "humor_tone":                       ["None","Light_Playful","Dark_Ironic","Sarcastic_Cynical"],
    "dialogue_prominence":              ["Silent_Visual_Driven","Low_Minimal","Moderate_Balanced","High_Dominant","Internal_Monologue_Reflection"],
    "sense_of_scale":                   ["Intimate_Personal","Small_Group_Local","Large_Group_Global","Vast_Epic_Cosmic"],
    "visual_pacing_style":              ["Slow_Pensive","Standard_Pacing","Action_Fast","Montage_Sequence","Abrupt_Shift","Chaotic_Disjointed","Stylized_Visuals"],
    "soundscape_elements":              ["Natural_Environment","Combat_Explosions","Mechanical_Industrial","Human_Vocalizations","Warning_Signals","Silence_Emphasis"],
    "thematic_elements":                ["Personal_Journey_Growth","Relationships_Human_Connection","Conflict_Societal_Struggle","Abstract_Conceptual"],
    "violence_level":                   (0, 10),
    "cultural_influence":               ["Noble_Formal","Indigenous_Spiritual","Ancient_Mystical","Imperial_Bureaucratic","Brutal_Oppressive",None],
    "mystical_type":                    ["None","Subtle_Dream","Vision_Prophecy","Ritual_Ceremony","Active_Power","Creature_Awe"],
    "spiritual_mystical_presence":      (1, 10),
    "technological_prominence":         (1, 10),
    "time_period_aesthetic":            ["Historical_Specific","Contemporary","Future_SciFi","Alternate_Dimension"],
    # SHOT-LEVEL CUE POINTS — Gemini identifies internal music shift moments within the scene
    # Always present; empty list [] when no internal shifts exist
    # Each entry: {"label": <short description>, "cue_type": <value>}
    "internal_cue_points":              "array_of_objects",  # validated separately, not via spec list
}

# Allowed values for internal_cue_points[].cue_type
CUE_POINT_TYPES = [
    "Tension_Peak",    # highest danger/dread moment in scene
    "Emotional_Break", # character breaks, cries, snaps
    "Horror_Reveal",   # something disturbing shown
    "Comedic_Beat",    # joke or gag lands
    "Silence_Beat",    # deliberate pause — anticipation or emptiness
    "Action_Beat",     # violence or physical action begins
    "Key_Line_Delivery",# thematically crucial line spoken
    "Monologue_Start", # character begins solo speech
    "Visual_Shift",    # POV, close-up, wide reveal, cutaway
    "Reality_Break",   # dream, memory, hallucination mid-scene
]

SCHEMA_TEXT = """
ANNOTATION SCHEMA — use ONLY these fields and values. NEVER invent new values.

ESSENTIAL (every scene):
  scene_id                          : integer
  scene_header                      : string (INT./EXT. line verbatim)
  narrative_stage                   : Beginning_Setup | Middle_Development | Climax_Turning_Point | End_Resolution | Specific_Narrative_Device
  emotional_core                    : Positive_Uplifting | Negative_Distressing | Neutral_Complex | Tension_Action | Awe_Wonder
  tension_level                     : 1-10
  pacing_intensity                  : 1-10
  scene_setting                     : Interior_Confined | Natural_Outdoor | Urban_Manmade | Military_Conflict_Zone | Abstract_Otherworldly
  character_focus                   : array of: Protagonist_Individual | Protagonist_Duo | Antagonist_Individual | Supporting_Character | Group_Collective | Neutral_Observer | Specific_Creature_Threat
  emotional_dramatic_shift_trigger  : true | false

HIGH IMPACT (null if not applicable):
  emotional_intensity               : 1-10
  action_intensity                  : 1-10
  conflict_nature                   : WHAT TYPE OF PRESSURE/DANGER exists in the scene:
                                      Physical_Danger | Psychological_Tension | Interpersonal_Conflict | Moral_Dilemma | Environmental_Threat | Unknown_Threat | Time_Pressure | Claustrophobic | null
  conflict_type                     : THE STRUCTURAL LEVEL of conflict (not the danger type):
                                      Internal_Personal | Interpersonal_Social | Large_Scale_Combat | Ideological_Political | Environmental_Survival | Existential_Abstract | None
  character_internal_state          : PICK EXACTLY ONE — do not combine values with underscores:
                                      Determined | Anxious | Confused | Angry | Sad | Calm | Focused | Vulnerable |
                                      Despairing | Triumphant | Disoriented | Suspicious | Manipulative | Manipulated |
                                      Self_Deceiving | Content | Curious | Defiant | Terrified | Conflicted |
                                      Frustrated | Shocked | Exhausted | Hopeful | Excited | Grieving |
                                      Resigned | Protective | Amazed | Ruthless | null
  character_transformation          : true | false
  reality_distortion_effect         : Present_Reality | Memory_Flashback | Dream_Fantasy | Internal_Monologue | Surreal_Psychedelic | Degraded_Fragmented | Bioluminescent_Ethereal | Stylized_Visuals
  memory_state_degradation          : Clear_Intact | Fading_Degraded | Fragmented_Confused | Traumatic_Distorted | Constructed_False | Desire_To_Forget | Reliance_External_Aids | null
  musical_cue_type                  : Original_Score_Underscore | Leitmotif_Thematic | Source_Music_Diegetic | Silence_Dramatic | Soundscape_Natural_Tribal | Genre_Specific_Performance

CONTEXTUAL (null if not applicable):
  foreshadowing_callback            : PICK FROM THIS LIST ONLY — do not write free text descriptions:
                                      None | Past_Trauma_Echo | Impending_Danger_Loss | Character_Arc_Development | Moral_Consequence | Prophecy_Legend | Symbolic_Echo | Thematic_Callback | null
  relationship_status               : Pre_Relationship | New_Potential | Developing_Attraction | Established_Intimate |
                                      Strained_Conflict | Strained_Alliance | Breakup_Separation | Post_Breakup_Recovery |
                                      Memory_Reflection | Developing_Mentorship | Antagonistic | Antagonistic_Rivalry |
                                      Open_Hostility | Alliance_Under_Pressure | Reluctant_Alliance | Mentor_Student | null
  symbolic_recurring_motif          : free-text string naming the specific motif (e.g. "Cultural_Clash", "Nature_vs_Industry") | null
  sense_of_repetition               : true | false
  moral_ambiguity                   : 1-10
  humor_tone                        : None | Light_Playful | Dark_Ironic | Sarcastic_Cynical
  dialogue_prominence               : Silent_Visual_Driven | Low_Minimal | Moderate_Balanced | High_Dominant | Internal_Monologue_Reflection
  sense_of_scale                    : Intimate_Personal | Small_Group_Local | Large_Group_Global | Vast_Epic_Cosmic
  visual_pacing_style               : Slow_Pensive | Standard_Pacing | Action_Fast | Montage_Sequence | Abrupt_Shift | Chaotic_Disjointed | Stylized_Visuals
  soundscape_elements               : array of: Natural_Environment | Combat_Explosions | Mechanical_Industrial | Human_Vocalizations | Warning_Signals | Silence_Emphasis
  thematic_elements                 : array of: Personal_Journey_Growth | Relationships_Human_Connection | Conflict_Societal_Struggle | Abstract_Conceptual
  violence_level                    : 0-10
  cultural_influence                : Noble_Formal | Indigenous_Spiritual | Ancient_Mystical | Imperial_Bureaucratic | Brutal_Oppressive | null
  mystical_type                     : None | Subtle_Dream | Vision_Prophecy | Ritual_Ceremony | Active_Power | Creature_Awe
  spiritual_mystical_presence       : 1-10
  technological_prominence          : 1-10
  time_period_aesthetic             : Historical_Specific | Contemporary | Future_SciFi | Alternate_Dimension

SHOT-LEVEL CUE POINTS (always include — empty array [] if no internal shifts):
  internal_cue_points               : array of objects, each with:
    label                           : short free-text description of the moment (max 6 words)
    cue_type                        : PICK EXACTLY ONE from:
                                      Tension_Peak | Emotional_Break | Horror_Reveal | Comedic_Beat |
                                      Silence_Beat | Action_Beat | Key_Line_Delivery |
                                      Monologue_Start | Visual_Shift | Reality_Break

  Only flag moments where the emotional register genuinely shifts.
  A flat dialogue scene → internal_cue_points: []
  Max 3 entries per scene — only the most significant shifts.
  Do NOT add an entry for every beat — only real emotional turning points.
"""

SYSTEM_PROMPT = """
You are an expert film music supervisor and score composer.
You annotate screenplay scenes for background music scoring.

Every annotation must reflect not just what happens in the scene right now,
but the full accumulated story state — character arcs, active tensions,
established motifs, relationship history, and audience emotional state.

The same scene scores completely differently depending on what came before.
Always use the provided movie_state when choosing field values.

STRICT RULE: Only include characters that actually appear in the scene text.
Never invent or hallucinate characters not present in the scene.

OUTPUT STYLE — CRITICAL:
- Return ONLY the JSON object. No explanations, no commentary, no preamble.
- Every field value must be a label or number — never a sentence or description.
- Be precise and concise. One correct value is better than a verbose wrong one.
- Do not justify your choices inside the JSON.
""".strip()

EMPTY_CONTEXT = {
    "narrative_state": {
        "stage": "Beginning_Setup",
        "plot_milestones": [],
        "audience_emotional_state": "neutral",
        "dominant_tone_so_far": "Neutral_Complex"
    },
    "tension_state": {
        "active_conflicts": [],
        "resolved_conflicts": [],
        "current_tension_trajectory": "neutral",
        "peak_tension_so_far": 0,
        "trajectory_history": []
        # each entry: {"after_scene": <id>, "trajectory": "<value>", "peak": <int>}
        # built locally in Python — never returned by Gemini
    },
    "character_states": {},
    # per character structure:
    # "Cole": {
    #   "current_emotion": "Anxious",        <- Gemini updates (scalar)
    #   "arc_position":    "doubting_reality",<- Gemini updates (scalar)
    #   "recent_key_event":"saw 1917 photo",  <- Gemini updates (scalar)
    #   "trauma_active":   true,              <- Gemini updates (scalar)
    #   "moral_stance":    "...",             <- Gemini updates (scalar)
    #   "transformation_occurred": false,     <- Gemini updates (scalar)
    #   "emotional_arc": [],  <- built locally: [{"after_scene":X,"emotion":"Y"}]
    #   "key_events":    []   <- built locally: [{"scene":X,"event":"<5 words max>"}]
    # }
    "relationship_states": {},
    # per pair structure:
    # "Cole_Railly": "reluctant_allies"       <- Gemini updates (scalar)
    # "Cole_Railly_history": []               <- built locally:
    #                                            [{"after_scene":X,"state":"Y"}]
    "motif_state": {
        "established_motifs": [],
        "active_foreshadowing": [],
        "resolved_foreshadowing": [],
        # built locally when item leaves active_foreshadowing:
        # [{"resolved_at_scene": X, "item": "..."}]
        "recurring_patterns": []
    },
    "moral_state": {
        "ambiguity_level": 1,
        "decisions_with_consequence": []
    },
    "memory_state": {
        "degradation_level": "Clear_Intact",
        "memory_events": [],
        "degradation_history": []
        # built locally: [{"after_scene": X, "level": "..."}]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# SAFE STATE MERGE — LLM cannot overwrite or lose previous memory
# ─────────────────────────────────────────────────────────────────────────────
def merge_state(old: dict, new: dict) -> dict:
    """
    Recursively merge new state into old state.
    - dicts   : recurse
    - lists   : union (old + new, deduplicated by string value)
    - scalars : take new value (LLM updated it intentionally)
    Old entries are NEVER deleted.
    """
    if not new:
        return old

    merged = old.copy()

    for key, new_val in new.items():
        old_val = merged.get(key)

        if isinstance(new_val, dict) and isinstance(old_val, dict):
            merged[key] = merge_state(old_val, new_val)

        elif isinstance(new_val, dict) and old_val is None:
            merged[key] = new_val

        elif isinstance(new_val, list) and isinstance(old_val, list):
            # FIX: active_foreshadowing — let Gemini's version win outright
            # It intentionally prunes items when they resolve; union-merge undoes that
            if key == "active_foreshadowing":
                merged[key] = new_val
            else:
                # Union of lists — keep all old, add new unique entries
                seen = set()
                combined = []
                for item in old_val + new_val:
                    key_str = json.dumps(item, sort_keys=True)
                    if key_str not in seen:
                        seen.add(key_str)
                        combined.append(item)
                merged[key] = combined

        elif isinstance(new_val, list) and old_val is None:
            merged[key] = new_val

        else:
            if key == "peak_tension_so_far":
                merged[key] = max(old_val or 0, new_val or 0)
            else:
                merged[key] = new_val

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY BUILDER — runs locally after each batch, zero API cost
# Watches what changed in movie_state scalars and appends to history lists
# Gemini never returns history — you own it, you build it, you send it as input
# ─────────────────────────────────────────────────────────────────────────────
def update_history(context: dict, old_state: dict, new_state: dict, last_scene_id: int) -> dict:
    """
    Compare old_state vs new_state after each batch.
    Append changes to history lists inside context.
    Called AFTER merge_state — context already has new scalar values.

    History entries are kept ULTRA SHORT — 5 words max per event.
    This keeps input tokens from growing.
    """
    def short(text: str, max_words: int = 5) -> str:
        """Truncate to max_words to keep token cost minimal."""
        if not text:
            return ""
        words = str(text).split()
        return " ".join(words[:max_words])

    # ── Character histories ────────────────────────────────────────────────
    old_chars = old_state.get("character_states", {})
    new_chars = new_state.get("character_states", {})

    for char_name, new_data in new_chars.items():
        if not isinstance(new_data, dict):
            continue

        old_data = old_chars.get(char_name, {})

        ctx_char = context.setdefault("character_states", {}).setdefault(char_name, {})
        ctx_char.setdefault("emotional_arc", [])
        ctx_char.setdefault("key_events", [])

        old_emotion = old_data.get("current_emotion")
        new_emotion = new_data.get("current_emotion")
        if new_emotion and new_emotion != old_emotion:
            ctx_char["emotional_arc"].append({
                "after_scene": last_scene_id,
                "emotion": new_emotion
            })

        new_event = new_data.get("recent_key_event", "")
        if new_event:
            # FIX: compare against last entry in the list, not old_data snapshot
            # old_data is pre-merge; Gemini echoes old text forward slightly rephrased
            existing_events = ctx_char["key_events"]
            last_event_text = existing_events[-1]["event"] if existing_events else ""
            # Simple similarity: skip if new event shares >60% words with last entry
            new_words  = set(short(new_event, 6).lower().split())
            last_words = set(last_event_text.lower().split())
            overlap    = len(new_words & last_words) / max(len(new_words), 1)
            if overlap < 0.6:
                ctx_char["key_events"].append({
                    "scene": last_scene_id,
                    "event": short(new_event, 6)
                })

    # ── Tension trajectory history ─────────────────────────────────────────
    old_traj = old_state.get("tension_state", {}).get("current_tension_trajectory")
    new_traj = new_state.get("tension_state", {}).get("current_tension_trajectory")
    if new_traj and new_traj != old_traj:
        context.setdefault("tension_state", {}) \
               .setdefault("trajectory_history", []) \
               .append({
                   "after_scene": last_scene_id,
                   "trajectory": new_traj,
                   "peak": context["tension_state"].get("peak_tension_so_far", 0)
               })

    # ── Relationship history ───────────────────────────────────────────────
    old_rels = old_state.get("relationship_states", {})
    new_rels = new_state.get("relationship_states", {})
    for pair, new_rel_state in new_rels.items():
        # FIX: skip _history keys Gemini echoes back — prevents _history_history
        if pair.endswith("_history"):
            continue
        old_rel_state = old_rels.get(pair)
        if new_rel_state and new_rel_state != old_rel_state:
            history_key = f"{pair}_history"
            context.setdefault("relationship_states", {}) \
                   .setdefault(history_key, []) \
                   .append({
                       "after_scene": last_scene_id,
                       "state": short(new_rel_state, 4)
                   })

    # ── Resolved foreshadowing ─────────────────────────────────────────────
    old_active = set(old_state.get("motif_state", {}).get("active_foreshadowing", []))
    new_active = set(new_state.get("motif_state", {}).get("active_foreshadowing", []))
    resolved   = old_active - new_active
    for item in resolved:
        context.setdefault("motif_state", {}) \
               .setdefault("resolved_foreshadowing", []) \
               .append({
                   "resolved_at_scene": last_scene_id,
                   "item": short(item, 6)
               })

    # ── Memory degradation history ─────────────────────────────────────────
    old_deg = old_state.get("memory_state", {}).get("degradation_level")
    new_deg = new_state.get("memory_state", {}).get("degradation_level")
    if new_deg and new_deg != old_deg:
        context.setdefault("memory_state", {}) \
               .setdefault("degradation_history", []) \
               .append({
                   "after_scene": last_scene_id,
                   "level": new_deg
               })

    return context


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_FIELDS = {
    "scene_id", "scene_header", "narrative_stage", "emotional_core",
    "tension_level", "pacing_intensity", "scene_setting",
    "character_focus", "emotional_dramatic_shift_trigger"
}

ARRAY_FIELDS = {"character_focus", "soundscape_elements", "thematic_elements", "internal_cue_points"}


def validate_annotation(ann: dict, scene_text: str) -> tuple[dict, float, list]:
    """
    Validate and clean one annotation against schema spec.
    Returns (cleaned_annotation, confidence_score, issues_list)
    """
    issues  = []
    cleaned = dict(ann)
    total   = len(SCHEMA_SPEC)
    valid   = 0
    missing_required = 0

    for field, spec in SCHEMA_SPEC.items():
        val = cleaned.get(field)

        if field not in cleaned or cleaned[field] is None:
            if field in REQUIRED_FIELDS:
                issues.append(f"MISSING required: {field}")
                cleaned[field] = None
                missing_required += 1
            else:
                cleaned[field] = None
                valid += 1
            continue

        # ── spec = None → free-text field (symbolic_recurring_motif) ──────────
        if spec is None:
            # Accept any non-empty string or null — normalise to snake_case style
            if isinstance(val, str) and val.strip():
                # Normalise: strip whitespace, replace spaces with underscores
                normalised = val.strip().replace(" ", "_").replace(".", "")
                cleaned[field] = normalised
                valid += 1
            else:
                cleaned[field] = None
                valid += 1
            continue

        if isinstance(spec, tuple):
            lo, hi = spec
            if not isinstance(val, (int, float)):
                try:
                    val = int(float(str(val)))
                    cleaned[field] = val
                    issues.append(f"TYPE COERCED {field}: string cast to {val}")
                except (ValueError, TypeError):
                    default = (lo + hi) // 2
                    issues.append(f"TYPE ERROR {field}: '{cleaned[field]}' unrecoverable — defaulted to {default}")
                    cleaned[field] = default
                    val = default
            if not (lo <= val <= hi):
                clamped = max(lo, min(hi, int(val)))
                issues.append(f"OUT OF RANGE {field}: {val} → clamped to {clamped}")
                cleaned[field] = clamped
                valid += 1
            else:
                valid += 1

        elif isinstance(spec, list):
            if field in ARRAY_FIELDS:
                if not isinstance(val, list):
                    val = [val] if val else []
                    cleaned[field] = val
                    issues.append(f"FIXED {field}: wrapped scalar in list")
                bad   = [v for v in val if v not in spec]
                good  = [v for v in val if v in spec]
                if bad:
                    issues.append(f"REMOVED invalid values from {field}: {bad}")
                    cleaned[field] = good
                valid += 1
            else:
                # ── Compound value guard ─────────────────────────────────────
                # Gemini sometimes combines values: "Anxious_Confused",
                # "Strained_Alliance_Developing", "Open_Hostility_Rivalry_Combat"
                # Try splitting on underscore and pick first valid token
                COMPOUND_FIELDS = {"character_internal_state", "relationship_status"}
                if field in COMPOUND_FIELDS and isinstance(val, str) and val not in spec:
                    # Build all possible prefix combinations (longest first)
                    tokens = val.split("_")
                    fixed  = None
                    # Try progressively longer prefixes first (e.g. "Open_Hostility" before "Open")
                    for length in range(len(tokens), 0, -1):
                        candidate = "_".join(tokens[:length])
                        if candidate in spec:
                            fixed = candidate
                            break
                    if fixed:
                        issues.append(
                            f"COMPOUND {field}: '{val}' → kept '{fixed}'"
                        )
                        cleaned[field] = fixed
                        valid += 1
                        continue
                    # No valid prefix found — set null
                    issues.append(f"INVALID {field}: '{val}' not in allowed values → set null")
                    cleaned[field] = None
                    continue

                if val not in spec:
                    issues.append(f"INVALID {field}: '{val}' not in allowed values → set null")
                    cleaned[field] = None
                else:
                    valid += 1

    # ── Validate internal_cue_points separately (array of objects) ──────────────
    raw_cps = cleaned.get("internal_cue_points")
    if raw_cps is None:
        cleaned["internal_cue_points"] = []
    elif not isinstance(raw_cps, list):
        issues.append("FIXED internal_cue_points: not a list → set []")
        cleaned["internal_cue_points"] = []
    else:
        valid_cps = []
        for cp in raw_cps:
            if not isinstance(cp, dict):
                issues.append(f"REMOVED invalid cue_point entry (not a dict): {cp}")
                continue
            cue_type = cp.get("cue_type")
            if cue_type not in CUE_POINT_TYPES:
                issues.append(f"INVALID cue_point cue_type: '{cue_type}' → removed entry")
                continue
            label = str(cp.get("label", "")).strip()
            if not label:
                label = cue_type  # fallback to type name if label missing
            valid_cps.append({
                "label": label,
                "cue_type": cue_type,
            })
        # Cap at 3 — more than 3 cue points per scene is noise
        cleaned["internal_cue_points"] = valid_cps[:3]

    scene_lower = scene_text.lower() if scene_text else ""

    if cleaned.get("character_focus"):
        roles = cleaned["character_focus"] if isinstance(cleaned["character_focus"], list) else []
        named_roles = [r for r in roles if r not in (
            "Group_Collective", "Neutral_Observer", "Specific_Creature_Threat"
        )]
        if named_roles and len(scene_lower.strip()) < 30:
            issues.append(
                f"POSSIBLE HALLUCINATION: character roles {named_roles} set "
                f"but scene text is very short — verify manually"
            )

    penalty    = missing_required * 2
    confidence = round(max(0.0, (valid - penalty) / total), 2)

    return cleaned, confidence, issues


# ─────────────────────────────────────────────────────────────────────────────
# ROBUST SCENE PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_scenes(script_text: str) -> list:
    """
    Splits screenplay into scenes on INT./EXT. headers.
    Handles all common variations:
      INT. BAR - NIGHT            standard with dot
      EXT JUNGLE - DAY            no dot
      INT./EXT. CAR - MOVING      combined
      I/E VEHICLE - NIGHT         shorthand
      INT. BAR - NIGHT            em-dash / em-dash
      EXT: STREET - DAY           colon separator
      175  INT. SITTING ROOM       scene number prefix (Apocalypse Now style)
      A1   EXT. FIELD - DAY        alphanumeric scene number prefix

    Fixes applied:
      - Normalise \r\n and \r to \n before splitting (Windows scripts)
      - Minimum length dropped from 50 to 20 chars (short scenes were silently dropped)
    """
    # Normalise line endings — \r\n and bare \r both become \n
    script_text = script_text.replace("\r\n", "\n").replace("\r", "\n")

    SPLIT_RE  = re.compile(
        r'(?=\n[ \t]*(?:[A-Za-z]?\d+\w*[ \t]+)?(?:INT\.?\/EXT\.?|INT\.?|EXT\.?|I\/E)(?:[\s\.:\-\u2013\u2014/]|$))',
        re.IGNORECASE
    )
    HEADER_RE = re.compile(
        r'^(?:[A-Za-z]?\d+\w*[ \t]+)?(?:INT\.?\/EXT\.?|INT\.?|EXT\.?|I\/E)(?:[\s\.:\-\u2013\u2014/]|$)',
        re.IGNORECASE
    )
    def is_valid_header(header: str) -> bool:
        """
        Extra guard beyond regex — rejects dialogue false positives like
        'Int. Who's your commanding' by checking that the location part
        is predominantly uppercase (real headers always are).
        """
        clean = re.sub(r'^[A-Za-z]?\d+\w*\s+', '', header).strip()
        clean = re.sub(r'^(?:INT\.?\/EXT\.?|INT\.?|EXT\.?|I\/E)[\s\.]*', '', clean, flags=re.IGNORECASE).strip()
        if not clean:
            return True  # bare INT./EXT. with no location — valid
        letters = [c for c in clean if c.isalpha()]
        if not letters:
            return True
        return (sum(1 for c in letters if c.isupper()) / len(letters)) >= 0.5

    parts    = SPLIT_RE.split(script_text)
    scenes   = []
    scene_id = 1
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        lines  = stripped.splitlines()
        header = lines[0].strip()
        if not HEADER_RE.match(header):
            continue
        if not is_valid_header(header):
            continue
        # Lowered from 50 to 20 — short scenes like "EXT. COPTER\nIt banks left."
        # were silently dropped at 50, causing gaps in scene_id numbering
        if len(stripped) > 20:
            scenes.append({"scene_id": scene_id, "header": header, "text": stripped})
            scene_id += 1
    return scenes


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(scenes_text: str,
                 scene_ids: list,
                 context: dict,
                 is_first_batch: bool,
                 full_script: str = "") -> str:

    script_block = (
        f"=== FULL SCRIPT ===\n{full_script}\n\n"
        if is_first_batch else ""
    )

    context_block = (
        "No prior context — first batch. Build movie_state from these scenes."
        if is_first_batch
        else f"=== CURRENT MOVIE STATE ===\n{json.dumps(context, indent=2, default=str)}"
    )

    return f"""
{script_block}{context_block}

=== SCHEMA ===
{SCHEMA_TEXT}

=== TASK ===
Annotate scenes {scene_ids[0]} to {scene_ids[-1]} below.

ANNOTATION RULES:
- Use movie_state context when choosing EVERY field value
- tension_level: use trajectory_history to judge if tension is building, peaked, or falling
- character_internal_state: use emotional_arc to understand the full journey to this moment
  CRITICAL: pick EXACTLY ONE value — never combine e.g. "Anxious_Confused" is INVALID
- emotional_core + character_internal_state: when the scene contains strong internal_cue_points
  (e.g. Horror_Reveal, Tension_Peak, Emotional_Break), let those moments nudge your emotion
  choice — they signal what emotionally defines the scene. Do not let a single cue_point
  override the scene's overall tone if the rest of the scene contradicts it.
- conflict_nature: the TYPE OF PRESSURE in the scene (Physical_Danger, Psychological_Tension etc)
  conflict_type: the STRUCTURAL LEVEL (Internal_Personal, Large_Scale_Combat etc) — these are different fields
- foreshadowing_callback: MUST be one of the allowed values — NEVER write free text like "Danger on Pandora"
  Use Impending_Danger_Loss for general danger, Symbolic_Echo for motif callbacks, Thematic_Callback for theme callbacks
- sense_of_repetition: only true if pattern already in recurring_patterns
- ONLY include characters that APPEAR IN THE SCENE TEXT — never invent characters
- confidence_score: your self-assessed accuracy 0.0-1.0

TOKEN BUDGET RULES — CRITICAL:
- In updated_movie_state, write SHORT values only
- recent_key_event: max 6 words (e.g. "escaped asylum with Railly's help")
- plot_milestones entries: max 6 words each (e.g. "Cole declared insane by doctors")
- active_conflicts entries: max 8 words each
- arc_position: max 4 words (e.g. "doubting_own_reality")
- moral_stance: max 4 words (e.g. "ends_justify_means")
- relationship state values: max 4 words (e.g. "reluctant_allies_with_tension")
- Do NOT write sentences or paragraphs anywhere in updated_movie_state
- Every value must be a label, not a description

=== SCENES TO ANNOTATE ===
{scenes_text}

=== OUTPUT FORMAT ===
Return ONLY valid JSON. No markdown. No backticks.

{{
  "annotations": [
    {{
      "scene_id": <int>,
      "scene_header": "<verbatim header>",
      "scene_text": "<full scene text verbatim>",
      "confidence_score": <0.0-1.0>,
      "narrative_stage": "...",
      "emotional_core": "...",
      "tension_level": <1-10>,
      "pacing_intensity": <1-10>,
      "scene_setting": "...",
      "character_focus": ["..."],
      "emotional_dramatic_shift_trigger": <true|false>,
      "emotional_intensity": <1-10>,
      "action_intensity": <1-10>,
      "conflict_nature": "...",
      "conflict_type": "...",
      "character_internal_state": "...",
      "character_transformation": <true|false>,
      "reality_distortion_effect": "...",
      "memory_state_degradation": "...",
      "musical_cue_type": "...",
      "foreshadowing_callback": "...",
      "relationship_status": "...",
      "symbolic_recurring_motif": "...",
      "internal_cue_points": [
        {{"label": "...", "cue_type": "..."}}
      ],
      "sense_of_repetition": <true|false>,
      "moral_ambiguity": <1-10>,
      "humor_tone": "...",
      "dialogue_prominence": "...",
      "sense_of_scale": "...",
      "visual_pacing_style": "...",
      "soundscape_elements": ["..."],
      "thematic_elements": ["..."],
      "violence_level": <0-10>,
      "cultural_influence": "...",
      "mystical_type": "...",
      "spiritual_mystical_presence": <1-10>,
      "technological_prominence": <1-10>,
      "time_period_aesthetic": "..."
    }}
  ],
  "updated_movie_state": {{
    "narrative_state": {{
      "stage": "<current stage>",
      "plot_milestones": ["<ALL previous>", "<new ones>"],
      "audience_emotional_state": "<current>",
      "dominant_tone_so_far": "<current>"
    }},
    "tension_state": {{
      "active_conflicts": ["<ALL unresolved>"],
      "resolved_conflicts": ["<ALL resolved>"],
      "current_tension_trajectory": "escalating|de-escalating|peaked|neutral|volatile",
      "peak_tension_so_far": <highest seen across ALL scenes>
    }},
    "character_states": {{
      "<Name>": {{
        "arc_position": "...",
        "current_emotion": "...",
        "recent_key_event": "...",
        "trauma_active": <true|false>,
        "moral_stance": "...",
        "transformation_occurred": <true|false>
      }}
    }},
    "relationship_states": {{
      "<CharA_CharB>": "<current state>"
    }},
    "motif_state": {{
      "established_motifs": ["<ALL previous>", "<new>"],
      "active_foreshadowing": ["<ALL unresolved>", "<new>"],
      "recurring_patterns": ["<ALL previous>", "<new>"]
    }},
    "moral_state": {{
      "ambiguity_level": <1-10>,
      "decisions_with_consequence": ["<ALL previous>", "<new>"]
    }},
    "memory_state": {{
      "degradation_level": "<current>",
      "memory_events": ["<ALL previous>", "<new>"]
    }}
  }}
}}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI CALLER
# ─────────────────────────────────────────────────────────────────────────────
def call_gemini(model, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err:
                wait = 20 * (attempt + 1)
                print(f"    Rate limit. Waiting {wait}s... ({attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after retries")


# FIX: strip code fences + escape illegal control characters in string values
def parse_json(raw: str) -> dict:
    raw   = raw.strip()
    raw   = re.sub(r'```(?:json)?', '', raw)   # strip Gemini code fences
    start = raw.find('{')
    end   = raw.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError('No JSON found in response')
    payload = raw[start:end]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Gemini sometimes embeds literal control characters inside string values
        # e.g. raw newlines in scene_text — replace with escaped versions
        _CTRL = {chr(10): r'\n', chr(13): r'\r', chr(9): r'\t'}
        payload = re.sub(r'[\x00-\x1f\x7f]',
                         lambda m: _CTRL.get(m.group(), ''), payload)
        return json.loads(payload)


def setup_model(script_text: str):
    try:
        cache_content = (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== FULL SCRIPT ===\n{script_text}\n\n"
            f"=== SCHEMA ===\n{SCHEMA_TEXT}"
        )
        cache = genai.caching.CachedContent.create(
            model=MODEL,
            display_name="annotator_cache",
            system_instruction=SYSTEM_PROMPT,
            contents=[cache_content],
            ttl=f"{CACHE_TTL_SECONDS}s"
        )
        model = genai.GenerativeModel.from_cached_content(
            cached_content=cache,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )
        print(f"    Cache created (~{len(cache_content)//4:,} tokens cached)")
        return model, cache
    except Exception as e:
        print(f"    Cache unavailable ({e}) — standard model")
        model = genai.GenerativeModel(
            model_name=MODEL,
            system_instruction=SYSTEM_PROMPT,  # FIX: was missing in fallback path
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )
        return model, None


# ─────────────────────────────────────────────────────────────────────────────
# CORE ANNOTATOR
# ─────────────────────────────────────────────────────────────────────────────
def annotate_script(script_path: str, output_path: str):

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    with open(script_path, "r", encoding="utf-8") as f:
        script_text = f.read()

    script_name  = os.path.basename(script_path).replace(".txt", "")
    scenes       = parse_scenes(script_text)
    total_scenes = len(scenes)

    if total_scenes == 0:
        print(f"  ERROR: No scenes found. Check script format.")
        return

    scenes_per_batch = max(5, OUTPUT_TOKEN_LIMIT // TOKENS_PER_SCENE)
    total_batches    = math.ceil(total_scenes / scenes_per_batch)

    print(f"\n{'='*60}")
    print(f"  Script  : {script_name}")
    print(f"  Scenes  : {total_scenes}")
    print(f"  Batches : {total_batches}  ({scenes_per_batch} scenes/batch)")
    print(f"{'='*60}")

    model, cache    = setup_model(script_text)
    using_cache     = cache is not None
    all_annotations = []
    context         = json.loads(json.dumps(EMPTY_CONTEXT))
    is_first_batch  = True
    total_issues    = 0

    # ── Scene-level progress bar ───────────────────────────────────────────
    pbar = tqdm(
        total      = total_scenes,
        desc       = f"  {script_name[:30]}",
        unit       = " scene",
        bar_format = "  {l_bar}{bar}| {n_fmt}/{total_fmt} scenes [{elapsed}<{remaining}, {rate_fmt}]",
        ncols      = 80
    )

    for batch_idx in range(total_batches):
        s_idx     = batch_idx * scenes_per_batch
        e_idx     = min(s_idx + scenes_per_batch, total_scenes)
        batch     = scenes[s_idx:e_idx]
        scene_ids = [s["scene_id"] for s in batch]

        scenes_text = "\n\n---\n\n".join(
            f"SCENE {s['scene_id']}:\n{s['text']}" for s in batch
        )

        ctx_size = len(json.dumps(context)) // 4
        pbar.write(
            f"\n  ┌─ Batch {batch_idx+1}/{total_batches} "
            f"| scenes {scene_ids[0]}–{scene_ids[-1]} "
            f"| ctx ~{ctx_size} tok "
            f"| {'📦 cached' if using_cache else '📄 full script'}"
        )
        pbar.set_description(f"  Batch {batch_idx+1}/{total_batches} — calling Gemini...")

        prompt = build_prompt(
            scenes_text    = scenes_text,
            scene_ids      = scene_ids,
            context        = context,
            is_first_batch = is_first_batch,
            full_script    = script_text if (is_first_batch and not using_cache) else ""
        )

        try:
            pbar.set_description(f"  Batch {batch_idx+1}/{total_batches} — waiting for Gemini...")
            raw    = call_gemini(model, prompt)

            pbar.set_description(f"  Batch {batch_idx+1}/{total_batches} — parsing response...")
            result = parse_json(raw)

            raw_annotations  = result.get("annotations", [])
            updated_state    = result.get("updated_movie_state", {})

            # Validate every annotation
            pbar.set_description(f"  Batch {batch_idx+1}/{total_batches} — validating...")
            batch_issues = 0
            for ann in raw_annotations:
                scene_txt = ann.get("scene_text", "")
                cleaned, confidence, issues = validate_annotation(ann, scene_txt)
                cleaned["confidence_score"] = confidence
                if issues:
                    cleaned["validation_issues"] = issues
                    batch_issues += len(issues)
                all_annotations.append(cleaned)

            total_issues += batch_issues

            # Strip hallucinated characters before merging
            if updated_state and "character_states" in updated_state:
                batch_text_lower = scenes_text.lower()
                filtered_chars = {}
                hallucinated = []
                for char_name, char_data in updated_state["character_states"].items():
                    if char_name.lower() in batch_text_lower:
                        filtered_chars[char_name] = char_data
                    elif char_name in context.get("character_states", {}):
                        filtered_chars[char_name] = char_data
                    else:
                        hallucinated.append(char_name)
                if hallucinated:
                    pbar.write(f"  ⚠️  HALLUCINATION REMOVED: {hallucinated}")
                    total_issues += len(hallucinated)
                updated_state["character_states"] = filtered_chars

            # Save old state before merging (needed for history diffing)
            old_state = json.loads(json.dumps(context))

            # Safe merge — LLM cannot overwrite previous memory
            if updated_state:
                context = merge_state(context, updated_state)

            # Build history locally by diffing old vs new state
            last_scene_id = scene_ids[-1]
            context = update_history(context, old_state, updated_state, last_scene_id)

            new_ctx_size = len(json.dumps(context)) // 4
            avg_conf     = round(
                sum(a.get("confidence_score", 0) for a in all_annotations[-len(raw_annotations):])
                / max(len(raw_annotations), 1), 2
            )

            n_chars    = len(context.get("character_states", {}))
            n_tensions = len(context.get("tension_state", {}).get("active_conflicts", []))
            n_motifs   = len(context.get("motif_state", {}).get("established_motifs", []))
            conf_icon  = "🟢" if avg_conf >= 0.8 else "🟡" if avg_conf >= 0.6 else "🔴"

            pbar.write(
                f"  └─ ✅ {len(raw_annotations)} scenes annotated "
                f"| {conf_icon} conf={avg_conf} "
                f"| issues={batch_issues} "
                f"| ctx={new_ctx_size} tok "
                f"| 👤 {n_chars} chars  ⚡ {n_tensions} tensions  🎵 {n_motifs} motifs"
            )

            # Advance progress bar by number of scenes just processed
            pbar.update(len(raw_annotations))
            is_first_batch = False

            # ── Incremental save after every batch ────────────────────────
            # If the run crashes mid-way, already-processed scenes are safe.
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            partial_output = {
                "script":             script_name,
                "total_scenes":       total_scenes,
                "annotated":          len(all_annotations),
                "batches_done":       batch_idx + 1,
                "batches_total":      total_batches,
                "avg_confidence":     round(
                    sum(a.get("confidence_score", 0) for a in all_annotations)
                    / max(len(all_annotations), 1), 2
                ),
                "total_issues_fixed": total_issues,
                "final_movie_state":  context,
                "annotations":        all_annotations
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(partial_output, f, indent=2, ensure_ascii=False)
            pbar.write(f"  💾 Saved {len(all_annotations)} scenes → {os.path.basename(output_path)}")

        except Exception as e:
            pbar.write(f"  └─ ❌ BATCH FAILED: {e}")
            pbar.write(f"       Saving partial results and continuing...")
            try:
                preview = raw[:600].replace('\n', ' ') if raw else '<empty response>'
                pbar.write(f"       RAW RESPONSE: {preview}")
            except NameError:
                pbar.write("       RAW RESPONSE: <not available>")
            pbar.update(len(batch))   # still advance so bar doesn't stall

        if batch_idx < total_batches - 1:
            for remaining in range(CALL_DELAY, 0, -1):
                pbar.set_description(f"  Cooldown {remaining}s before next batch...")
                time.sleep(1)

    pbar.set_description(f"  ✅ {script_name[:30]} — done")
    pbar.close()

    if cache:
        try:
            cache.delete()
            print(f"\n  Cache deleted.")
        except Exception:
            pass

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    avg_confidence = round(
        sum(a.get("confidence_score", 0) for a in all_annotations)
        / max(len(all_annotations), 1), 2
    )

    # Final write — updates batches_done to equal batches_total
    # and overwrites the last incremental save with the clean final version
    output = {
        "script":             script_name,
        "total_scenes":       total_scenes,
        "annotated":          len(all_annotations),
        "batches_done":       total_batches,
        "batches_total":      total_batches,
        "avg_confidence":     avg_confidence,
        "total_issues_fixed": total_issues,
        "final_movie_state":  context,
        "annotations":        all_annotations
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Saved           : {os.path.abspath(output_path)}")
    print(f"  Annotated       : {len(all_annotations)}/{total_scenes}")
    print(f"  Avg confidence  : {avg_confidence}")
    print(f"  Issues fixed    : {total_issues}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        print("  export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    if not os.path.isdir(SCRIPTS_DIR):
        print(f"ERROR: '{SCRIPTS_DIR}/' not found.")
        sys.exit(1)

    scripts = sorted([f for f in os.listdir(SCRIPTS_DIR) if f.endswith(".txt")])
    if not scripts:
        print(f"ERROR: No .txt files in {SCRIPTS_DIR}/")
        sys.exit(1)

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Movie Script Annotator")
    print(f"  Input  : {os.path.abspath(SCRIPTS_DIR)}/")
    print(f"  Output : {os.path.abspath(ANNOTATIONS_DIR)}/")
    print(f"{'='*60}\n")

    pending = []
    for fname in scripts:
        out  = os.path.join(ANNOTATIONS_DIR, fname.replace(".txt", ".json"))
        done = os.path.exists(out)
        print(f"  {'[SKIP]' if done else '[    ]'} {fname}")
        if not done:
            pending.append(fname)

    if not pending:
        print(f"\n  All done. Delete from {ANNOTATIONS_DIR}/ to re-annotate.\n")
        sys.exit(0)

    print(f"\n  {len(pending)} script(s) to annotate.\n")

    for i, fname in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] 🎬 {fname}")
        annotate_script(
            os.path.join(SCRIPTS_DIR, fname),
            os.path.join(ANNOTATIONS_DIR, fname.replace(".txt", ".json"))
        )
        if i < len(pending) - 1:
            print(f"\n  ⏳ Pausing 15s before next script...")
            for remaining in range(15, 0, -1):
                print(f"\r     {remaining}s remaining...  ", end="", flush=True)
                time.sleep(1)
            print()

    done_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".json")]
    print(f"\n{'='*60}")
    print(f"  ALL DONE — {len(done_files)} file(s) in {ANNOTATIONS_DIR}/")
    print(f"{'='*60}\n")