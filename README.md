# CineEmotion: Multi-Stage Cascading Architecture for Film Music Description

> **From Scene Text to Orchestration** — Predicting film music descriptors from raw screenplay text using a three-stage cascading neural pipeline.

---

## Overview

CineEmotion processes raw screenplay scenes through three progressively sophisticated modules to predict concrete music parameters including tempo, tonality, harmonic style, dynamics, rhythm, texture, and orchestration — all from text alone.
Evaluated on **81 annotated films** (~11,000 scenes), ScoreForge achieves:
- Tempo prediction **R² = 0.842** (MAE = 7.28 BPM)
- Musical valence **R² = 0.863**
- Tonality classification **86.1% accuracy**

---

## Repository Structure
CineEmotion/
│
├── final_module1.ipynb       # Module 1 — Scene Perception (training + evaluation)  
├── final_2.ipynb             # Module 2 — Narrative Context (training + evaluation)  
├── final_3.ipynb             # Module 3 — Music Descriptor Prediction (training + evaluation)  
├── inference.ipynb           # End-to-end inference on new screenplay scenes  
│  
└── README.md  

---

## Pipeline Architecture

### Module 1 — Scene Perception (`final_module1.ipynb`)
- **Backbone**: `distilroberta-base` (768-d), fine-tuned
- **Input**: Raw scene text (max 512 tokens)
- **Output**: 256-d scene embedding + 11 predictions

| Head | Type | Output |
|---|---|---|
| emotional_valence | 4-class | Positive / Neutral / Tension / Negative |
| conflict_nature | 6-class | Physical / Psychological / Interpersonal / ... |
| acoustic_space | 6-class | Interior_Small / Outdoor_Natural / ... |
| reality_layer | 5-class | Present / Memory / Dream / ... |
| score_dynamic_shape | 4-class | Build_Release / Sustained / ... |
| scene_interaction_tone | 5-class | Conflict / Bonding / Expository / ... |
| pacing_intensity | regression | 1–10 |
| action_intensity | regression | 0–10 |
| scene_tension_raw | regression | 1–10 |
| scene_arousal | regression | 0–1 |
| emotion_tags | 7-label | Anger / Joy / Sadness / Fear / ... |

### Module 2 — Narrative Context (`final_2.ipynb`)
- **Architecture**: 4-layer, 8-head Pre-LN Transformer Encoder
- **Input**: 5-scene sliding window of 304-d feature vectors from M1
- **Output**: 256-d context vector + 6 predictions

| Head | Type | Output |
|---|---|---|
| tension_level | regression | 1–10 |
| arousal_level | regression | 1–10 |
| emotional_shift_trigger | binary | True / False |
| narrative_arc_position | 5-class | Setup / Rising / Climax / Falling / Resolution |
| foreshadowing_type | 4-class | None / Foreshadow / Payoff / Echo |
| transition_type | 5-class | attacca / fade / segue / silence / cut |

### Module 3 — Music Descriptor Prediction (`final_3.ipynb`)
- **Architecture**: MLP on 314-d input (41-d M1 labels + 16-d M2 labels + 256-d context vector)
- **Input**: Ground truth M1/M2 labels during training; predicted labels at inference
- **Output**: 8 music descriptor predictions

| Head | Type | Output |
|---|---|---|
| tempo_bpm | regression | 45–170 BPM |
| musical_valence | regression | −1.0 to +1.0 |
| tonality | 3-class | atonal / major / minor |
| harmonic_style | 7-class | chromatic / diatonic / modal / ... |
| dynamic_shape_m4 | 8-class | crescendo / sustained / swell / ... |
| rhythm_style | 6-class | drive / pulse / rubato / sparse / ... |
| texture | 5-class | ambient / chamber / full / hybrid / solo |
| orchestration | 14-label | strings / percussion / piano / brass / ... |

---

## Results

### Module 1
| Head | Accuracy | F1-Macro |
|---|---|---|
| emotional_valence | 0.559 | 0.510 |
| acoustic_space | 0.767 | 0.612 |
| scene_interaction_tone | 0.568 | 0.428 |
| pacing_intensity | MAE=1.02 | R²=0.557 |
| action_intensity | MAE=1.25 | R²=0.603 |

### Module 2
| Head | Score |
|---|---|
| tension_level | MAE=1.21, R²=0.565 |
| arousal_level | MAE=1.20, R²=0.559 |
| emotional_shift (F1) | 0.524 (recall=0.729) |
| narrative_arc (Acc) | 0.605 |

### Module 3
| Head | Score |
|---|---|
| tempo_bpm | MAE=7.28, **R²=0.842** |
| musical_valence | MAE=0.128, **R²=0.863** |
| tonality | **Acc=0.861**, F1-Mac=0.714 |
| dynamic_shape_m4 | Acc=0.774, F1-Mac=0.607 |
| orchestration | F1-Mac=0.274, F1-Wt=0.629 |

---

## Inference (`inference.ipynb`)

To run inference on your own screenplay scenes:

```python
# Input format — list of scene dicts
scenes = [
    {
        "scene_id": 1,
        "scene_header": "EXT. RAINFOREST - DAY",
        "scene_text": "EXT. RAINFOREST - DAY\nThe sun's rays shaft down..."
    },
    {
        "scene_id": 2,
        "scene_header": "INT. LINK ROOM",
        "scene_text": "INT. LINK ROOM\nJake sits in a chair, talking straight to camera..."
    }
]
```

The inference notebook:
1. Downloads M1 checkpoint from HuggingFace → runs scene perception
2. Downloads M2 checkpoint from HuggingFace → runs narrative context
3. Downloads M3 checkpoint from HuggingFace → predicts music descriptors
4. Prints per-scene music predictions in a readable table

---

## Pretrained Models

All checkpoints are publicly available on HuggingFace:

| Module | Model Card |
|---|---|
| M1 — Scene Perception | [suyashnpande/scene-perception-m1-harshal](https://huggingface.co/suyashnpande/scene-perception-m1-harshal) |
| M2 — Narrative Context | [suyashnpande/narrative-context-m2-harshal](https://huggingface.co/suyashnpande/narrative-context-m2-harshal) |
| M3 — Music Descriptors | [suyashnpande/music-descriptor-m3](https://huggingface.co/suyashnpande/music-descriptor-m3) |

---

## Dataset

- **81 annotated films**, ~11,120 scenes
- Each scene annotated with M1, M2, and M3 labels
- **Split**: 64 train / 8 validation / 8 test films (film-level, seed=42)
- No scene from a test film appears in training

---

## Setup

```bash
pip install torch transformers huggingface_hub scikit-learn numpy
```

All notebooks are designed to run on **Kaggle** with GPU enabled. Set your HuggingFace tokens as Kaggle Secrets:
- `HF_READ_TOKEN`
- `HF_WRITE_TOKEN`

---

## Citation
