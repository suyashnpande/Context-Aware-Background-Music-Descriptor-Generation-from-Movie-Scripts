# Context-Aware Background Music Descriptor Generation from Movie Scripts

This project presents a hierarchical deep learning pipeline that converts movie or TV screenplay dialogue into structured background music descriptors. The system analyzes emotional signals, scene dynamics, and narrative structure to generate musically meaningful attributes such as tempo, harmony, orchestration, and dynamic shape.

The goal is to bridge natural language understanding and computational music generation by modeling how film composers interpret scripts when designing background scores.

---

## Key Idea

Film music is strongly influenced by narrative context. Composers typically analyze scripts by understanding:

- Character emotions
- Scene-level dynamics
- Narrative tension
- Story arc progression

This project models these aspects using a **multi-stage deep learning architecture**.

---

## System Architecture

The pipeline consists of four modules:

### Module 1 – Dialogue Emotion Encoder
A transformer-based model processes dialogue utterances and produces contextual emotional embeddings.

Output:
- 256-dimensional emotion vector per utterance

---

### Module 2 – Scene Encoder
A bidirectional LSTM models emotional trajectories across utterances within a scene and produces a scene-level representation.

Outputs include:
- Scene vectors
- Dominant scene emotion
- Emotion transitions
- Scene intensity signals

---

### Module 3 – Narrative LSTM
A causal sequence model processes scene vectors across an episode to capture narrative progression and tension dynamics.

Additional mechanisms:
- **TensionMemoryCell** for modeling dramatic pressure
- **CharacterMemoryBank** for tracking emotional states of characters

Outputs:
- Narrative vectors
- Arc phase (setup, rising, climax, falling, resolution)
- Narrative tension scores

---

### Module 4 – Music Planner
A multi-task neural network converts narrative features into structured music descriptors.

Predicted attributes include:
- Tempo
- Tonality
- Harmonic style
- Motion type
- Dynamic shape
- Texture
- Instrumentation

These descriptors resemble the cues used in professional film scoring.

---

## Dataset

The system uses the **MELD dataset**, derived from the TV show *Friends*.

Dataset statistics:

- ~13,700 dialogue utterances
- 1,337 scenes
- 180 episodes
- 7 emotion classes

Scenes are constructed by grouping dialogue segments within each episode.

---

## Results

Performance across modules:

| Module | Task | Macro F1 |
|------|------|------|
| M1 | Dialogue emotion recognition | 31.95% |
| M2 | Scene-level emotion modeling | 55.32% |
| M3 | Narrative arc prediction | 52.47% |
| M4 | Music descriptor prediction | **70.04%** |

The results show that hierarchical narrative modeling significantly improves music descriptor prediction.

---

## Project Highlights

- Hierarchical multi-level architecture (dialogue → scene → narrative)
- Self-supervised labeling using film-scoring heuristics
- Multi-task learning across 13 musical attributes
- ~9.6M parameter deep learning system

---

## Applications

Potential applications include:

- AI-assisted film scoring
- Script-aware music generation
- Intelligent soundtrack recommendation
- Narrative-driven music composition systems

---

## Repository Structure
