# Paper Implementations & Discussions

Papers are where most of this stack came from, and they're also where most of us stop reading. You skim the abstract, nod at the headline number, save the PDF, and never find out whether the idea holds up. This section is the antidote: pick a paper, rebuild its core idea small enough to run on a laptop, and see for yourself.

Status: **planned.** The folder exists, the shape is decided, no paper is in yet.

---

## Why this section exists

The rest of the repo teaches the settled parts of the stack — tokenization, attention, RAG, fine-tuning. Settled ideas are easier to explain because someone already did the work of boiling them down. Papers haven't been boiled down yet. That's exactly why they're worth the effort: reading one carefully and reproducing it is the fastest way to tell a real advance from a good chart.

It's also the honest test of everything in `00-Foundations`. If you can read a new attention variant and implement it from the maths, the foundations stuck. If you can't, you know which notebook to go back to.

---

## What a paper folder looks like

One folder per paper, named for the idea rather than the title:

```
11-Paper_Implementations_and_Discussions/
  00-Some_Idea/
    README.md            — the paper, the claim, what we reproduce and what we skip
    00-Introduction.ipynb — the reproduction
```

The notebook follows the same contract as every other notebook here (see [AUTHORING_GUIDE.md](../AUTHORING_GUIDE.md)) with three additions:

- **Link the paper up front** — title, authors, arXiv ID, date. The reader should be able to open it alongside the notebook.
- **Reproduce the claim, not the benchmark.** A paper's headline number usually needs a cluster. Shrink the setup until it runs on CPU, then check the *direction* of the result holds. Say out loud what you shrank.
- **End with the discussion.** What does this change for someone building things? Is it already in the libraries you use? What would you want to test before trusting it?

No paper gets marked done because the code ran. It gets marked done when the notebook states plainly whether the idea reproduced, partly reproduced, or didn't.

---

## Picking papers

Bias towards papers that are (a) recent enough to still be arguable, (b) small enough that the core idea fits in one notebook, and (c) load-bearing — something a working engineer would actually change their approach over. A clever result that needs 64 GPUs to see is a bad fit here, however good the paper is.

---

Next up: nothing yet. When the first paper lands it gets a row in the root [README](../README.md) table like every other subsection.
