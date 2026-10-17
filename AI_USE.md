# AI_USE.md — AI Assistance Disclosure

In compliance with the IncuBrix Engineering Assessment 3 guidelines, this document fully discloses all AI coding assistants, models, and tools used during the design, implementation, and benchmarking of this system.

---

## 1. AI Coding Assistants & Tools

*   **Antigravity (Powered by Google DeepMind / Gemini)**:
    *   *Usage*: End-to-end architecture design, scaffolding of FastAPI routes, FFmpeg filtergraph construction for smart 9:16 blurred background crop, drafting unit tests for hard cases, and building the responsive HTML5/CSS3 frontend.
    *   *Portion of Codebase*: Provided initial code generation, refactoring, and integration test setup; reviewed and verified line-by-line.

---

## 2. AI Models Embedded in the Application

*   **Faster-Whisper (`whisper-base`)**:
    *   *Role*: Automatic speech recognition (ASR) with word-level forced alignment to accurately determine scene boundaries and subtitle timestamps.
*   **Neural Edge-TTS Voices**:
    *   *Role*: Text-to-speech synthesis for multilingual voiceovers (`en-US-AriaNeural`, `hi-IN-SwaraNeural`, `ta-IN-PallaviNeural`).
*   **Vision-Language Model (VLM) Metric Evaluator**:
    *   *Role*: Evaluation of scene relevance scores by comparing video frames against spoken clauses.
