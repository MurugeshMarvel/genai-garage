# Authoring Guide

Everything in this repo is a notebook, and every notebook follows the same shape. This guide is
that shape, written down. Read it before you open a PR — it's short, and following it is the
difference between a contribution that merges in an hour and one that needs three rounds of
review.

If you've never contributed here, jump to [Your first contribution](#your-first-contribution).

---

## What this repo is trying to do

Most GenAI material stops at the analogy. You get a tidy picture of attention as "the model
paying attention", and then it moves on before anything is concrete.

So the promise here is narrow and specific: **every concept gets code you can run.** Built from
scratch in numpy or torch first, then compared to the library version once the mechanics are
clear. One everyday analogy so the idea has a hook. At least one plot so you can see it happen.

Three principles fall out of that, and they decide most review comments:

- **Honest over impressive.** If the code is an approximation, the notebook says so. Never hide a
  step to make the output look cleaner.
- **From scratch first.** Reaching for the library before building the thing defeats the point.
- **Runnable beats readable.** A cell that only runs on your machine is a bug.

Readers are assumed to know Python — functions, loops, pip. They are *not* assumed to know ML
theory, maths notation, or to own a GPU.

---

## Getting set up

```bash
git clone https://github.com/MurugeshMarvel/genai-garage.git
cd genai-garage

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # add ".[rag]" if you're working on RAG notebooks

jupyter lab
```

Foundations notebooks need no API keys — they're pure numpy/torch. Anything that calls a model
uses the `garage_helper` router, which will walk you through credentials the first time:

```python
from garage_helper import setup_llm, LLMRouter

MODEL  = setup_llm()          # loads .env, or asks you for a provider and key
router = LLMRouter(default_model=MODEL)

print(router.generate("Explain attention in one sentence."))
```

Pick whichever provider you already have a key for — OpenAI, Azure, Anthropic, Bedrock, Gemini,
DeepSeek, or a local Ollama. See [`garage_helper/README.md`](./garage_helper/README.md) for the
full list.

**Never import `openai`, `anthropic`, or `boto3` directly in a notebook.** Everything goes
through the router, so a reader with a different provider can still run your work.

And never commit a key, an ARN, an endpoint, or an account ID. Those live in `.env`, which is
gitignored.

---

## How the repo is laid out

Numbered **sections** at the top level, numbered **subsections** inside them, one notebook per
subsection:

```
00-Foundations/
  00-Tokenization/
    00-Introduction.ipynb
    README.md
  01-Embeddings/
    ...
```

Conventions:

- Folders are `NN-Title_Case_With_Underscores`. Two-digit prefix, always.
- The main notebook in a subsection is `00-Introduction.ipynb`. Demos can have descriptive names
  (`chat_with_pdf.ipynb`).
- No `&` in new folder names — it forces URL-encoding in every link. Use `_and_`.
- Every subsection has a `README.md` of roughly 150 words: the problem, the analogy, what you'll
  run.
- Order matters. Later notebooks lean on earlier ones, so don't insert a subsection that assumes
  something the reader hasn't met yet.

---

## The notebook contract

### Cell order

```
[markdown]  Title              — one H1
[code]      Install cell       — commented !pip install line for Colab
[markdown]  Recap + the problem — what the last notebook covered, then what breaks without this
[markdown]  Concept 1          — analogy first, then the explanation
[code]      Concept 1 demo     — minimal code that proves it
[markdown]  Concept 2          — analogy, then explanation
[code]      Concept 2 demo
...
[markdown]  Putting it together
[code]      Full example       — end to end, nothing hidden
[markdown]  ## Key takeaways   — 5–8 bullets, the last one bridges to the next notebook
```

One markdown cell per concept, one code cell per concept. Don't merge two concepts into one cell —
it's the single most common reason a PR gets sent back.

### Writing style

Write like a developer explaining something to a colleague, not like documentation.

- Plain, direct sentences. Active voice.
- Vary sentence length. Never three same-length sentences in a row.
- Contractions are fine and preferred — "it's", "you'll", "doesn't".
- Lead with the problem or a question. **Never open with a definition.**
- Use "you" and "we". The reader is a participant, not an observer.
- Don't write "In this notebook we will cover the following topics:" followed by a bullet list.
  Say the purpose in a sentence.
- Headings are descriptive phrases — "Softmax — turning scores into probabilities", not "Softmax".

### Analogies

Every concept needs one, before the code. It's the thing readers remember six months later.

- Draw from everyday life: food, sports, driving, parties, school, music, movies, cooking.
- Slightly funny or unexpected is better than clever.
- Signal it with `**The analogy:**` so it's easy to spot when skimming.
- The mapping has to actually hold. A half-fitting analogy is worse than none.
- One per concept. Don't stack three.

Already taken, so pick something else: softmax → lunch scores for biryani vs burger vs salad ·
residual connections → a telephone game with a direct phone line added · learning rate too high →
a guitar amp cranked to max · gradient descent → hiking in fog, feeling the slope underfoot ·
tokenization → teaching a child phonics · layer norm → a mixing engineer normalising each track ·
warmup schedule → motorway speed first, slow near the address · multi-head attention → a panel of
cooking-competition judges each scoring a different dimension.

### Code

- Must run in a fresh kernel with only the packages in the install cell.
- Keep cells under ~40 lines. Split if longer.
- Variable names are clear English: `similarity_scores`, not `s` or `result_1`.
- End each demo with one print that says what the output *means* in plain English.
- Never silence output. Every cell should show it ran.
- No cells that contain only imports — put imports at the top of the first cell that uses them.
- Align printed tables with f-string padding (`:<20`, `:>8`).
- Prefer a plot or table over a wall of numbers.

### Visualisations

At least one per notebook. Text-only notebooks aren't accepted.

- `matplotlib`, `figsize=(9,4)` or `(10,5)`. Wider is fine for comparisons.
- `plt.tight_layout()` before `plt.show()`.
- Label both axes and set a title. An unlabelled plot is noise.
- Colours: `steelblue` by default, `tomato` for errors or highlights, `seagreen` for success or
  comparison.
- After `plt.show()`, add 2–3 prints describing what the reader should notice.

### Continuity

Notebooks are read in order, so they need to connect.

- Open with a 2–3 sentence recap naming the previous notebook's topic. (Skip for the first
  notebook in a section.)
- Close with a bridge: `Next up: **Topic Name** — one sentence on why it comes next.` (Skip for
  the last.)

### Key takeaways

Every notebook ends with a `## Key takeaways` markdown cell:

- 5–8 bullets. No more.
- One sentence each, with the concept name bolded first:
  `- **Softmax** converts raw scores into a probability distribution…`
- The last bullet — or a line after a `---` — is the bridge to the next notebook.

### Notebook metadata

Use this block so notebooks stay diff-friendly:

```python
metadata = {
    "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.13.7"
    }
}
```

`nbformat: 4`, `nbformat_minor: 5`. Cell IDs are `"cell-0"`, `"cell-1"`, and so on.

---

## Rules per section

| Section | Extra rule |
|---|---|
| Foundations | Only `numpy`, `torch`, `matplotlib`, `sklearn`. No API calls. Build from scratch, then compare to the library version. |
| Prompting & Reasoning | Live API calls through `garage_helper`. Show the raw prompt, the response, and why it worked or failed. |
| Inference & Decoding | Show the probability distribution, not just the sampled token. Plot how it shifts as parameters change. |
| RAG Systems | All three phases every time — ingest, retrieve, generate. Print the retrieved chunks before the final answer. |
| Fine-Tuning | Before/after on the same prompt, base vs tuned. State dataset size and training time. |
| Agents & Tools | At least one multi-step reason → act → observe loop, with the full trace shown. |
| Multimodal | Display the actual image or audio input, not just its filename. |
| Evaluation | Aggregate into a scoring table. Never print individual scores alone. |
| Safety & Security | Show the attack and the mitigation side by side. Defensive framing only. |
| Deployment | Report real numbers — latency, tokens/sec, cost. |
| Observability | Show a real trace with token counts and cost attached. |

---

## Things we don't merge

- Explanations that open with a dictionary definition.
- Multi-paragraph docstrings inside notebook code cells.
- Comments that restate the code (`# create the model`).
- Filler like "Here we demonstrate the concept of X by doing Y". Just do Y.
- Variables named `data`, `result`, `output`, `temp`, `x1`, `x2`.
- Cells that only contain imports.
- Notebooks that end without a key takeaways cell.
- "As you can see from the output above…" — the output speaks for itself.
- Committed notebooks that were never executed.
- A section marked *built* when its notebooks haven't been run.

---

## Submitting a PR

1. Branch off `main`.
2. Run your notebook top to bottom in a fresh kernel and **commit the outputs.** GitHub renders
   them, so a reader sees your plots without running anything. An unexecuted notebook reads as
   abandoned.
3. Write or update the subsection `README.md`.
4. Add or update the row in the root `README.md` table, including the Colab badge:
   `https://colab.research.google.com/github/MurugeshMarvel/genai-garage/blob/main/<path>`
   The path is case-sensitive and must match the folder on disk.
5. Add any new dependency to both `pyproject.toml` and the notebook's install cell.
6. Open the PR describing what a reader will be able to do that they couldn't before.

Checklist before you hit submit:

- [ ] Runs top to bottom in a fresh kernel; outputs committed
- [ ] Every concept has an analogy
- [ ] At least one plot, with labelled axes and a title
- [ ] Key takeaways cell — 5–8 bullets, bridge included
- [ ] No keys, ARNs, endpoints, or absolute local paths anywhere
- [ ] Every relative link and Colab badge resolves
- [ ] New dependencies declared in both places

---

## Your first contribution

Good places to start, roughly easiest first:

- **Fix something wrong.** The whole point of this repo is that the code is honest, so a broken
  cell or an incorrect explanation is the most valuable thing you can report. Open an issue with
  the notebook path and what you saw.
- **Write a missing subsection README.** Around 150 words: the problem, the analogy, what the
  reader will run. Low risk, and it's what search engines actually index.
- **Add a visualisation** to a notebook that's light on them.
- **Write a new notebook** for a subsection folder that's still empty. Comment on the issue first
  so two people don't write the same one.
- **Add a provider** to `garage_helper`. Usually about seven lines — see the "Adding a New
  Provider" section of [`garage_helper/README.md`](./garage_helper/README.md).

Questions are welcome as issues. So is disagreeing with something in this guide — if a rule is
getting in the way of a clearer explanation, say so and we'll change the rule.
