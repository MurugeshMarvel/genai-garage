# My GenAI Garage

This is where I pull large language models apart to see how they actually work. Not slides, not an animated explainer, not a YouTube playlist I half-watch and forget. Code I can run, change, and break until the idea finally sticks.

Most GenAI material stops at the analogy. You get a tidy picture of attention as "the model paying attention," and then it moves on before anything is concrete. That never worked for me. So every topic here is a notebook with three parts: a short explanation, one real-world analogy to anchor it, and then the actual code that proves it. From scratch first with numpy or torch, then the library version once the mechanics are clear.

It starts at "how does text even become numbers" and builds up from there. You don't need a GPU or a PhD — just Python and some curiosity.

## How this repo is organised

Each numbered folder is a **section**. Sections break into **subsections**, and each subsection is a self-contained notebook you can open and run top to bottom. The order is deliberate: later notebooks lean on earlier ones, so working through a section in sequence is the intended path.

Some sections are finished, some are scaffolding, and some are still just a plan. I'd rather show the whole map and be honest about what's built than pretend it's all done. Status is marked plainly: **built**, **scaffolded** (folders exist, notebooks coming), or **planned**.

Every notebook follows the same shape, so you always know what you're walking into:

- **The problem first** — why this thing exists and what breaks without it.
- **An analogy** — something from everyday life so the idea has a hook.
- **The code** — minimal, runnable, nothing hidden.
- **Key takeaways** — the handful of things worth remembering, plus a pointer to what comes next.

---

## The sections

### Foundations — *built*

How text becomes numbers, and how a transformer actually learns, built piece by piece from scratch.
More: [00-Foundations/00-Foundations_Readme.md](./00-Foundations/00-Foundations_Readme.md)

| Subsection | Why it's here | Run |
|---|---|---|
| [00 · Tokenization](./00-Foundations/00-Tokenization/00-Introduction.ipynb) | How raw text gets chopped into tokens the model can read. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/genai-garage/blob/main/00-Foundations/00-Tokenization/00-Introduction.ipynb) |
| [01 · Embeddings](./00-Foundations/01-Embeddings/00-Introduction.ipynb) | How tokens turn into vectors that carry meaning. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/genai-garage/blob/main/00-Foundations/01-Embeddings/00-Introduction.ipynb) |
| [02 · Attention & Softmax](./00-Foundations/02-Attention%26Softmax/00-Introduction.ipynb) | How the model decides which words to focus on. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/genai-garage/blob/main/00-Foundations/02-Attention%26Softmax/00-Introduction.ipynb) |
| [03 · Transformer Internals](./00-Foundations/03-Transformer_Internals/00-Introduction.ipynb) | Positional encoding, residuals and layer norm assembled into a block. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/genai-garage/blob/main/00-Foundations/03-Transformer_Internals/00-Introduction.ipynb) |
| [04 · Learning Objectives](./00-Foundations/04-Learning_Objectives/00-Introduction.ipynb) | What the model is actually trained to predict. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/genai-garage/blob/main/00-Foundations/04-Learning_Objectives/00-Introduction.ipynb) |
| [05 · Training Mechanics](./00-Foundations/05-Training_Mechanics/00-Introduction.ipynb) | Loss, gradients, optimisers and the schedules that make training work. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/genai-garage/blob/main/00-Foundations/05-Training_Mechanics/00-Introduction.ipynb) |

### Prompting & Reasoning — *scaffolded*

How to steer a model with words alone, from a plain instruction all the way to getting it to reason step by step. Everything here changes the output without touching the model's weights.

Six of the nine notebooks are drafted and readable, but they're not finished: most have no saved outputs yet, and the Colab path needs work before the badges go up. Read them, don't rely on them.

| Subsection | Why it's here | Run |
|---|---|---|
| [00 · Anatomy of a Prompt](./01-Prompting_and_Reasoning/00-Anatomy_of_a_Prompt/00-Introduction.ipynb) | What a prompt really is to the model, and why wording changes everything. | draft |
| [01 · Zero-Shot Prompting](./01-Prompting_and_Reasoning/01-Zero_Shot_Prompting/00-Introduction.ipynb) | Getting good answers with clear instructions and no examples. | draft |
| [02 · Few-Shot Prompting](./01-Prompting_and_Reasoning/02-Few_Shot_Prompting/00-Introduction.ipynb) | Teaching a pattern by showing examples inside the prompt. | draft |
| [03 · Controlling Output](./01-Prompting_and_Reasoning/03-Controlling_Output/00-Introduction.ipynb) | Forcing the shape you need — structure, JSON, format, length. | draft |
| [04 · Chain of Thought](./01-Prompting_and_Reasoning/04-Chain_of_Thought/00-Introduction.ipynb) | Getting the model to show its working on multi-step problems. | draft |
| [05 · Advanced Reasoning](./01-Prompting_and_Reasoning/05-Advanced_Reasoning/00-Introduction.ipynb) | Self-consistency, decomposition and tree-of-thought. | draft |
| 06 · ReAct & Reflection | Interleaving reasoning with actions, plus self-critique loops. | planned |
| 07 · Test-Time Compute | Letting the model think longer, and when reasoning models beat prompting. | planned |
| 08 · Prompt Development & Robustness | The draft → test → refine workflow, and handling brittle prompts. | planned |

### Inference & Decoding — *planned*

How the model turns a pile of probabilities into actual text: sampling, temperature, top-p, and constrained/structured decoding. Subsections to be defined.

### RAG Systems — *scaffolded*

Give the model your own documents to answer from, instead of relying on whatever it memorised during training.

| Subsection | Why it's here | Run |
|---|---|---|
| [00 · RAG Foundations](./03-RAG_Systems/00-RAG_Foundations/rag_fundamentals.ipynb) | The three phases of RAG — ingest, retrieve, generate. | notes only, code coming |
| [01 · Simple RAG](./03-RAG_Systems/01-Simple_RAG/chat_with_pdf.ipynb) | A working chat-with-your-PDF demo, end to end. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurugeshMarvel/genai-garage/blob/main/03-RAG_Systems/01-Simple_RAG/chat_with_pdf.ipynb) |

### Agents & Tools — *planned*

Let the model take actions in a reason → act → observe loop instead of just answering. Subsections to be defined.

### Fine-Tuning — *planned*

Change the model's weights for your task, including alignment (instruction tuning, RLHF, DPO). Subsections to be defined.

### Multimodal — *planned*

Working with images, audio and text together. Subsections to be defined.

### Evaluation — *planned*

Measure whether the output is actually any good before you ship it — benchmarks, LLM-as-judge, regression sets. Subsections to be defined.

### Safety & Security — *planned*

Prompt injection, jailbreaks, guardrails, PII handling and red-teaming — keeping the thing from misbehaving. Subsections to be defined.

### Deployment Operation — *planned*

Serve and scale a model in production: inference servers, quantization, batching, latency and cost. Subsections to be defined.

### Observability — *planned*

Watch what's happening in production: tracing, token and cost tracking, drift detection, online eval. Subsections to be defined.

### Paper Implementations & Discussions — *planned*

Take a recent paper and reproduce its core idea in runnable code, with a short discussion of what it changes and why it matters. One folder per paper, each linking back to the original. Subsections to be defined.

---

## Running it

Everything is a Jupyter notebook. To run locally:

```bash
# 1. Clone
git clone https://github.com/MurugeshMarvel/genai-garage.git
cd genai-garage

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install what the notebooks use
pip install jupyter
pip install -e .                   # add ".[rag]" for the RAG notebooks

# 4. Launch
jupyter lab
```

The Foundations notebooks run with no API keys — they're pure numpy/torch. Anything that calls a model goes through the `garage_helper` router, which walks you through credentials the first time:

```python
from garage_helper import setup_llm, LLMRouter

MODEL  = setup_llm()          # loads .env, or asks you for a provider and key
router = LLMRouter(default_model=MODEL)
```

Bring whichever key you already have — OpenAI, Azure, Anthropic, Bedrock, Gemini, DeepSeek, or a local Ollama. Details in [garage_helper/README.md](./garage_helper/README.md).

Keep `.env` out of git (it's already in `.gitignore`). Never commit keys.

**Prefer Colab?** Hit the *Open In Colab* badge next to any built notebook above and it opens ready to run — just add your keys in the notebook for the sections that need them.

---

## A note on the state of things

This is a personal learning repo that grows whenever I dig into something new, so the map above will keep changing. New subsections get added under existing sections, and the paper section fills in as I work through papers worth reproducing. If a folder looks empty, it's on the list — not abandoned.

If you spot something wrong in a notebook (the whole point is that the code is honest), open an issue or a PR. Every notebook here follows one written-down shape — if you want to add or fix one, [AUTHORING_GUIDE.md](./AUTHORING_GUIDE.md) is that shape.
