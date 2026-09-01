# Multi-Judge Panel Extension for GSMA Open Telco Evals

This repository is a fork of [gsma-labs/evals](https://github.com/gsma-labs/evals)
(MIT licensed), used for the MSc thesis *"Domain-Adapted SLM-as-Judge for
Specialised Domains: A Human-Grounded Evaluation in Telecommunications."*
Everything from the `---` divider below is GSMA's original, unmodified
documentation.

## What's added

A multi-judge panel scoring layer for TeleQNA, TeleLogs (5G-Faults), and
TeleInter: simultaneous grading by multiple LLM judges and local SLM judges
(TSLAM-4B, Phi-4-Mini-Instruct), each producing a binary correctness grade
plus a 0–10 reasoning-quality score. Everything new lives in its own
`judge_panels/` directory — no existing GSMA file or folder is touched.

```
evals/
├── src/evals/judge_panels/         # new — all additions live here
│   ├── teleqna_multi_judge.py      # TeleQNA multi-judge scorer
│   ├── fiveG_faults_multi_judge.py # TeleLogs (5G-Faults) multi-judge scorer
│   └── teleinter_multi_judge.py    # TeleInter multi-judge scorer
└── .env.example                    # new: required environment variables
```

## Setup

Follow GSMA's own [Getting Started](docs/getting-started.md) guide to create a
`.env` file with `OPENROUTER_API_KEY`. Additionally, the local SLM judge
(TSLAM-4B) reads its checkpoint path from `TSLAM_4B_MODEL_PATH`; without it,
the default falls back to a placeholder path and will fail to load unless
overridden via `-T judge_models=[...]`.

## Running these evals

```
inspect eval src/evals/judge_panels/teleqna_multi_judge.py
inspect eval src/evals/judge_panels/fiveG_faults_multi_judge.py
inspect eval src/evals/judge_panels/teleinter_multi_judge.py
```

Override the judge panel via `-T judge_models=...` (either form works):

```
inspect eval src/evals/judge_panels/teleqna_multi_judge.py \
  -T judge_models="openrouter/openai/gpt-5.5,openrouter/google/gemini-3.1-pro-preview"

inspect eval src/evals/judge_panels/teleqna_multi_judge.py \
  -T judge_models="['openrouter/openai/gpt-5.5', 'hf/./models/tslam-4B']"
```

## Judge configuration

Each judge is called with `max_tokens=2048`, `temperature=0.1`, `verbosity="low"`.

- **Reasoning effort:** judges matching `gemini`, `o1`, `o3`, or `claude-3-7`/`claude-3.7`
  in their model id are called with `reasoning_effort="low"`; all others use `"none"`.
  The NVIDIA `openai-api/` endpoint rejects the `reasoning_effort` parameter outright,
  so it is omitted entirely for that provider (`teleqna_multi_judge.py` and
  `fiveG_faults_multi_judge.py` only — see below).
- **Malformed-output retries:** if a judge's response is missing a parseable `GRADE`
  or `reasoning_quality` field, the scorer re-prompts the same judge up to 3 attempts
  total, appending a corrective instruction on retries 2 and 3. The final attempt's
  output is used regardless of whether it parsed successfully; a still-unparsed grade
  defaults to `Incorrect`.
- **Transient-error retries:** independently of the malformed-output retry above,
  `max_retries=3` bounds `inspect_ai`'s own retry-on-transient-error behavior (timeouts,
  connection errors), so a persistently unreachable judge fails after 3 attempts
  instead of retrying indefinitely.
- **Request timeout:** `teleqna_multi_judge.py` and `fiveG_faults_multi_judge.py` set
  `client_timeout=30` (seconds) specifically for `openai-api/` judges, bounding the
  underlying HTTP request so an unresponsive endpoint cannot stall the run. This is
  **not** present in `teleinter_multi_judge.py`, which does not target the NVIDIA
  endpoint by default and uses a simpler, unconditional `reasoning_effort` config.

## About

This work benchmarks LLM-as-judge and SLM-as-judge grading reliability against
a 7-rater human-validation study across two telecom QA/troubleshooting
datasets (TeleQNA, TeleLogs), comparing a locally-hosted, domain-adapted SLM
judge (TSLAM-4B) against frontier LLM judges (GPT-5.5, Gemini-3.1-Pro,
Claude-Opus-4.8) and a general-purpose SLM baseline (Phi-4-Mini-Instruct). See
the accompanying thesis for the full methodology, results, and discussion of
judge-panel reliability, cost, and latency trade-offs.

For the base evaluation framework, see GSMA's own
[Getting Started](docs/getting-started.md) and
[Running Evaluations](docs/running-evaluations.md) guides below.

## License

MIT — inherited from the upstream [gsma-labs/evals](https://github.com/gsma-labs/evals)
repository (see `LICENSE`). The new files listed above are released under the
same license.

---------------------------------


<p align="center">
  <img src="docs/imgs/open_telco.svg" alt="GSMA Open_Telco" width="400">
</p>

# Open Telco

This repository is a suite of telco-specific benchmarks.

Our goal is to create a centralised hub where telco evaluations can be maintained and run locally.

📚 [Getting Started](docs/getting-started.md) · 🏃 [Running Evaluations](docs/running-evaluations.md) · 📋 [List of Evals](docs/eval-list.md) · 📝 [Blog Post](https://huggingface.co/blog/otellm/gsma-benchmarks-02)

We are particularly excited about developing evaluations that are realistic and address the complementary capabilities necessary to ensure safe and optimal deployment of AI in a telco environment.
If you share this mission, please [reach out](mailto:emolero@gsma.com), we are always looking for collaborators and contributors!

This project is built on [Inspect AI](https://inspect.aisi.org.uk/), we encourage everyone to familiarise themselves with the framework, it's rapidly becoming the standard evaluation framework across top AI research institutions.

## Collaborators

**Tech & Research:** GSMA, Huawei GTS, The Linux Foundation, Khalifa University, Universitat Pompeu Fabra (UPF), University of Texas, and Queen’s University.

**Telcos:** AT&T, China Telecom, Deutsche Telekom, du, KDDI, KPN, Liberty Global, Orange, Telefónica, Turkcell, Swisscom, Vodafone.

**Industry Labs & SMEs:** NetoAI, Datumo, Adaptive-AI
