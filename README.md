# Multi-Judge Panel Extension for GSMA Open Telco Evals

This repository is a fork of [gsma-labs/evals](https://github.com/gsma-labs/evals)
(MIT licensed), used for the MSc thesis *"Domain-Adapted SLM-as-Judge for
Specialised Domains: A Human-Grounded Evaluation in Telecommunications."*
Everything from the `---` divider below is GSMA's original, unmodified
documentation.

## What's added

A multi-judge panel scoring layer for TeleQNA and TeleLogs: simultaneous grading by
multiple LLM judges and local SLM judges (TSLAM-4B, Phi-4-Mini-Instruct), each
producing a binary correctness grade plus a 0–10 reasoning-quality score. No
existing GSMA file is modified.

```
evals/
├── src/evals/teleqna/
│   └── teleqna_mult_judge.py    # new: TeleQNA multi-judge scorer
├── src/evals/telelogs/
│   ├── telelogs_judge.py        # new: TeleLogs multi-judge scorer
│   └── teleinter_judge.py       # new: TeleInter multi-judge scorer
└── .env.example                 # new: required environment variables
```

## Setup

Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY`. The local SLM
judge (TSLAM-4B) reads its checkpoint path from `TSLAM_4B_MODEL_PATH`; without
it, the default falls back to a placeholder path and will fail to load unless
overridden via `-T judge_models=[...]`.

## Running these evals

```
inspect eval src/evals/teleqna/teleqna_mult_judge.py
inspect eval src/evals/telelogs/telelogs_judge.py
inspect eval src/evals/telelogs/teleinter_judge.py
```

Override the judge panel via `-T judge_models=...` (either form works):

```
inspect eval src/evals/teleqna/teleqna_mult_judge.py \
  -T judge_models="openrouter/openai/gpt-5.5,openrouter/google/gemini-3.1-pro-preview"

inspect eval src/evals/teleqna/teleqna_mult_judge.py \
  -T judge_models="['openrouter/openai/gpt-5.5', 'hf/./models/tslam-4B']"
```


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
