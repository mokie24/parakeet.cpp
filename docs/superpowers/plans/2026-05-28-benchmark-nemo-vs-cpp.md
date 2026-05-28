# parakeet.cpp — Benchmark (NeMo vs ggml) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Benchmark every released Parakeet model with NeMo (PyTorch, CPU) vs our C++/ggml engine — **speed** (RTFx/latency/thread-scaling/memory), **accuracy** (WER vs ground truth), and **agreement** (NeMo-vs-ours transcript WER) — plus a real-audio sanity check, with **plots** and a `BENCHMARK.md`.

**Approach:** A CLI `bench` subcommand times our engine cleanly (load once, transcribe a manifest, per-file timing). A Python runner drives NeMo (batch=1, matched thread count) and our `bench` over the same audio, computes metrics (normalized WER, RTFx = audio_sec/proc_sec, peak RSS via `/usr/bin/time`), and writes JSON. A plotting script renders matplotlib charts + `benchmarks/BENCHMARK.md`. Models are converted (f32 + q8_0) and benchmarked one at a time.

**Audio:**
- **LibriSpeech test-clean subset** (configurable N, default ~100 utts) with ground-truth refs → formal WER.
- **Diverse clips** (16 kHz mono): `jfk.wav` (whisper/LocalAI canonical), and from `~/_git/voxtral.c/samples/`: `I_have_a_dream.ogg` (MLK), `antirez_speaking_italian_short.ogg` (Italian → exercises multilingual `tdt-0.6b-v3`), `test_speech.wav`. Converted to 16k mono.

**Models:** all 10 validated — `parakeet-{ctc-0.6b,ctc-1.1b,rnnt-0.6b,rnnt-1.1b,tdt-0.6b-v2,tdt-0.6b-v3,tdt-1.1b,tdt_ctc-110m,tdt_ctc-1.1b}` + streaming `parakeet_realtime_eou_120m-v1`.

**Tech Stack:** C++ CLI, `.venv` NeMo (PyTorch CPU), matplotlib, ctest. 20 cores, ~130 GB free, `/usr/bin/time` available. Branch `bench/asr-benchmark`.

**Reference:** sibling bench scripts `/home/mudler/_git/rt-detr.cpp/scripts/{bench.py,bench_threads.py,plot_community.py}` + `benchmarks/` layout; our `parakeet-cli`, `scripts/validate_vs_nemo.py` (has a WER helper to reuse/centralize), `scripts/convert_parakeet_to_gguf.py`.

**Conventions:** ctest 0/77/fail; commit per green step; CPU-only; `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`. Benchmark audio + GGUFs are gitignored (`benchmarks/audio/`, `benchmarks/models/`); only scripts, results JSON, plots, and BENCHMARK.md are committed.

---

## Orientation + Entry gate
Read plan; verify env (`.venv` NeMo, build green — 35 tests). Resume via `git log`.

## Task 1: CLI `bench` subcommand (clean timing)
**Files:** `examples/cli/main.cpp`.
- [ ] Add `parakeet-cli bench --model m.gguf --manifest files.txt [--decoder ctc|tdt] [--threads N] [--json out.json]`: load the model ONCE; for each wav path in the manifest, load audio, time ONLY the transcription (`pk::Model::transcribe_path`), and record `{path, audio_sec, proc_ms, text}`. Emit JSON: `{"model":..,"threads":N,"load_ms":..,"files":[{path,audio_sec,proc_ms,text}, ...]}`. (`audio_sec` from the decoded sample count / 16000.) Set ggml threads from `--threads` (check how the existing graph runner picks thread count — wire `--threads` through to `pk::run_graph`/the CPU backend; if not currently configurable, add a thread param). Keep `info`/`transcribe`/`quantize`/`--stream` unchanged.
- [ ] Build; smoke `parakeet-cli bench --model /tmp/pk110m.gguf --manifest <(echo tests/fixtures/speech.wav) --json /tmp/b.json` → valid JSON with proc_ms. Commit: `git commit -m "feat: parakeet-cli bench subcommand (load-once per-file timing + JSON)"`.

## Task 2: Benchmark data prep
**Files:** `scripts/bench_data.py`, `.gitignore`.
- [ ] `scripts/bench_data.py --out benchmarks/audio`: (a) download a LibriSpeech **test-clean** subset (use `datasets`/`soundfile` or the openslr tar; take the first N=`--n 100` utterances), write each to `benchmarks/audio/librispeech/<id>.wav` (16k mono) + a manifest `benchmarks/librispeech_manifest.tsv` (`wav_path \t reference_text`). (b) Copy + convert the diverse clips to 16k mono wav under `benchmarks/audio/diverse/` (source `jfk.wav` from `/home/mudler/_git/voxtral.c/samples/jfk.wav`; `I_have_a_dream.ogg`, `antirez_speaking_italian_short.ogg`, `test_speech.wav` from the same dir — convert ogg→wav via `soundfile`/`librosa` resample to 16k mono) + a `benchmarks/diverse_manifest.tsv` (ref text empty/optional for clips without ground truth). Gitignore `benchmarks/audio/`.
- [ ] Run it (small N first to verify), confirm manifests + wavs exist + are 16k mono. Commit the script + `.gitignore` (NOT the audio): `git commit -m "feat: benchmark data prep (LibriSpeech subset + diverse clips)"`.

## Task 3: Benchmark runner (NeMo vs ours)
**Files:** `scripts/benchmark.py`, `scripts/asr_metrics.py` (shared normalized WER).
- [ ] `scripts/asr_metrics.py`: `normalize(text)` (lowercase, strip punctuation, collapse whitespace) + `wer(ref, hyp)` (word-level Levenshtein). Reuse in validate_vs_nemo.py if convenient.
- [ ] `scripts/benchmark.py --models <list|all> --manifest <librispeech|diverse> --threads 20 --out benchmarks/results/`: for each model:
  - Convert to GGUF (f32; also q8_0) via the converter; record GGUF sizes.
  - **NeMo** (PyTorch, CPU): `torch.set_num_threads(N)`; pick the head (transducer default, ctc for ctc models); transcribe each file **batch=1**, timing only the transcribe call; capture text + per-file proc time; capture peak RSS by running the NeMo pass as a subprocess under `/usr/bin/time -v` (or `resource.getrusage`).
  - **Ours**: run `parakeet-cli bench --model <f32 and q8_0> --manifest ... --threads N --json` under `/usr/bin/time -v` for peak RSS; parse the JSON timings/texts.
  - Compute per-engine: total RTFx (Σaudio_sec / Σproc_sec), median per-file latency, **WER vs ground truth** (LibriSpeech; normalized), **agreement WER** (NeMo vs ours), peak RSS, GGUF/checkpoint size.
  - Write `benchmarks/results/<model>.json`. After each model, optionally prune its HF cache + GGUFs (disk; we have headroom so optional).
  - **Thread sweep** (a small flag, e.g. `--thread-sweep 1,2,4,8,20` on ONE model over the diverse clips) → `benchmarks/results/threads.json`.
- [ ] Run on the 110m first (fast) to validate end-to-end; confirm RTFx/WER/agreement/mem populate sanely (agreement WER should be ~0 — we match NeMo). Commit: `git commit -m "feat: benchmark runner (NeMo vs ggml: RTFx/WER/agreement/mem) + WER metric"`.

## Task 4: Plots + BENCHMARK.md
**Files:** `scripts/plot_benchmark.py`, `benchmarks/BENCHMARK.md`.
- [ ] `scripts/plot_benchmark.py --results benchmarks/results --out benchmarks/plots`: matplotlib charts (PNG): (1) **RTFx per model**, NeMo vs ours (grouped bars; f32 + q8_0); (2) **speedup factor** ours/NeMo; (3) **WER per model** (ours vs NeMo vs—where applicable—ground truth); (4) **latency vs audio length** (scatter, diverse clips); (5) **thread scaling** (RTFx vs threads); (6) **peak memory** per model. Clean styling (titles, axis labels, legend).
- [ ] `benchmarks/BENCHMARK.md`: methodology (machine, cores, NeMo version, audio sets, batch=1, matched threads), a results table per model (size, RTFx NeMo/ours/speedup, WER ground-truth NeMo/ours, agreement WER, peak RSS), the embedded plots, and the diverse-clip transcripts (NeMo vs ours side by side) for the real-audio sanity check.
- [ ] Generate plots + draft BENCHMARK.md from the 110m results to validate the pipeline. Commit: `git commit -m "feat: benchmark plots + BENCHMARK.md generator"`.

## Task 5: Full run + write-up
**Files:** `benchmarks/results/*.json`, `benchmarks/plots/*.png`, `benchmarks/BENCHMARK.md`.
- [ ] Run the full benchmark across ALL 10 models on the LibriSpeech subset + diverse clips (this is long — NeMo CPU on the 1.1B models is slow; run in the background). Generate the plots + finalize `BENCHMARK.md` with real numbers. Sanity-check: agreement WER ≈ 0 for every model (confirms fidelity at scale on real audio); flag any model where ours diverges from NeMo on real audio (would be a real finding). Note the multilingual case: `tdt-0.6b-v3` on the Italian clip.
- [ ] Commit results + plots + BENCHMARK.md: `git commit -m "bench: full NeMo-vs-ggml results, plots, and write-up"`.

---

## Done-when
- `parakeet-cli bench` emits clean per-file timings; runner produces per-model JSON (RTFx, WER, agreement, mem) for all 10 models on LibriSpeech + diverse clips; plots + `BENCHMARK.md` generated with real numbers.
- Real-audio sanity: agreement WER ≈ 0 across models (our transcripts match NeMo on real audio, not just the parity fixtures); diverse-clip transcripts recorded; any divergence flagged.
- Full suite still green (the `bench` subcommand didn't regress anything).
