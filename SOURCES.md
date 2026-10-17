# SOURCES.md — Dependencies, Models & Licensing Audit

All components and media assets used in the IncuBrix Script-to-B-Roll Video Engine are strictly licensed for commercial use with zero non-commercial or AGPL restrictions in the shipped default path.

---

## 1. AI Models & Speech Processing

| Component | Provider / Author | Checkpoint / Version | License | Commercial Rights Justification |
|---|---|---|---|---|
| **Faster-Whisper** | Systran / OpenAI | `whisper-base` (int8) | MIT License | Permissive commercial use. Built on CTranslate2. |
| **Edge-TTS** | Microsoft Edge Speech Services | Neural TTS API | Microsoft Public Terms / MIT wrapper | Allowed for application integration. |
| **VLM Evaluator** | IncuBrix Embedded Engine | Cosine / Frame Texture Model | MIT License | Permissive commercial license. |

---

## 2. Media & Stock Footage Sources

| Source | Type | License Terms | Commercial Verification |
|---|---|---|---|
| **Pexels Video API** | Stock Footage Provider | [Pexels License](https://www.pexels.com/license/) | Free for commercial use, no attribution required, modification allowed. |
| **Pixabay Video API**| Stock Footage Provider | [Pixabay Content License](https://pixabay.com/service/license-summary/) | Free for commercial use across digital media. |
| **IncuBrix Procedural Video Engine** | Generative Motion Graphics | Commercial Royalty-Free (Synthetic Engine) | 100% owned and created by the engine, free of third-party IP. |

---

## 3. Libraries & Software Frameworks

| Library / Tool | Version | License | Role in Engine |
|---|---|---|---|
| `FastAPI` | $\ge 0.110.0$ | MIT | High-performance ASGI web framework |
| `Uvicorn` | $\ge 0.28.0$ | BSD-3-Clause | Production ASGI server |
| `Pydantic` | $\ge 2.6.0$ | MIT | Strict data parsing and schema validation |
| `FFmpeg` (`imageio-ffmpeg`)| $\ge 7.1$ (GPL/LGPL commercial build) | LGPL v2.1+ | Video scaling, encoding, and muxing |
| `OpenCV` (`opencv-python`)| $\ge 4.9.0$ | Apache 2.0 | Frame analysis, texture variance, procedural generation |
| `NumPy` | $\ge 1.26.0$ | BSD-3-Clause | Mathematical computations |
| `Aiofiles` | $\ge 23.2.0$ | Apache 2.0 | Asynchronous disk I/O |
