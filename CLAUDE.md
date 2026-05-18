# CLAUDE.md — RADVOX

> Herda as convenções globais de `C:/projetos/CLAUDE.md`.

## Objetivo
Aplicação de transcrição de voz para radiologistas. Converte laudos ditados em texto estruturado usando ASR (faster-whisper / Whisper local) + LLMs (OpenAI, Ollama, Gemini) com templates customizáveis.

## Status
**ARQUIVADO** — desenvolvimento migrado para VOXRAD2 (previsto mid-2026). Manutenção corretiva apenas; sem novas features.

## Stack Específica
- **ASR:** faster-whisper (local), OpenAI Whisper API
- **LLMs:** OpenAI API, Ollama (local), Google Gemini (`google-generativeai`)
- **Áudio:** sounddevice, soundfile, pydub, lameenc
- **UI:** pynput, pyautogui
- **Entry point:** `VoxRad.py`
- **Config:** `config/`
- **Templates:** `config/templates/`

## Regras Específicas
- Status arquivado: apenas bug fixes — não implementar features novas
- API keys (OpenAI, Google) ficam em `.env` — nunca no código ou `config/`
- Modelos Ollama rodam localmente: documentar o modelo exato usado em cada template
- Dados de áudio de pacientes: nunca commitar, nunca logar
- Mudanças em templates de laudo exigem revisão por um radiologista (fora do escopo da IA)

## O que NÃO fazer
- Não adicionar novas integrações de LLM (projeto arquivado)
- Não expor dados de transcrição em logs de debug
- Não hardcodar nomes de modelos Ollama — usar config
