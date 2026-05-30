# Local STT and MedGemma Integration Plan

## Goal
Implement a fully local, offline pipeline that transcribes audio dictation in Portuguese using Whisper and generates structured clinical reports/impressions using a quantized Portuguese clinical model.

## Tasks
- [x] Task 1: Check hardware and Ollama status → Verify: run `ollama --version` and `nvidia-smi` to confirm setup.
- [x] Task 2: Configure local defaults in `settings.ini` → Verify: run `python setup_local_env.py` and inspect `settings.ini`.
- [x] Task 3: Pull local models (Whisper and quantized Clinical-BR) → Verify: run `ollama pull hf.co/QuantFactory/Clinical-BR-LlaMA-2-7B-GGUF:Q4_K_M`.
- [x] Task 4: Implement local speech-to-text in `audio/transcriber.py` → Verify: run `python verify_asr.py` and check local transcription output.
- [x] Task 5: Connect local LLM prompt template in `llm/format.py` → Verify: format dummy transcribed text and check if `IMPRESSÃO DIAGNÓSTICA` is generated.

## Done When
- [x] Local audio dictation is transcribed locally using faster-whisper.
- [x] Transcribed text is successfully processed by a local quantized model.
- [x] A structured clinical report in Portuguese with findings and impressions is generated without external network calls.

## Notes
- Whisper V3 Large Turbo requires ~6GB VRAM; Llama-2-7B Q4_K_M requires ~4.1GB VRAM.
- Total VRAM footprint is around ~10GB, fitting comfortably within the RTX 5060 Ti (16GB VRAM).
