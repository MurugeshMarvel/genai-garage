# Foundations

Before you can build anything meaningful with AI, you need to understand what is actually happening under the hood. Not at a research-paper level, but enough to know why things work, why they sometimes fail, and how to make better decisions when building.

That is what this section is for.

---

## Why Foundations Matter

Most GenAI tutorials skip straight to the API call. You type a question, the model returns an answer, and it feels like magic. The problem is that when something goes wrong — the model gives a wrong answer, your search returns irrelevant results, your fine-tuned model performs worse than expected — you have no way to diagnose it.

Understanding the foundations turns that magic into engineering.

**A simple example:** Imagine you are building a chatbot that searches through your company's documents. You notice it keeps returning the wrong paragraphs. Without foundations, you can only guess. With foundations, you know exactly where to look — was the text chunked poorly? Are the embeddings not capturing the right meaning? Is the similarity calculation being done correctly? You can pinpoint the problem and fix it.

---

## What This Section Covers

Each topic builds on the previous one. The order is intentional.

### Tokenization — *How does text become numbers?*

AI models do not read words. They read numbers. Tokenization is the process of converting text into small pieces — called tokens — and then mapping those pieces to numbers the model can process.

This is the very first step in every language model pipeline. If you do not understand tokenization, you will not understand why some prompts cost more than others, why certain languages are handled differently, or why splitting text in the wrong place breaks a model's understanding.

### Embeddings — *How do numbers carry meaning?*

Once text is converted to numbers, those numbers need to carry semantic meaning. Embeddings are dense numerical representations where similar meanings end up close together in space.

Think of it like a map where "dog" and "puppy" are placed next to each other, while "dog" and "skyscraper" are placed far apart. This is what allows an AI to understand that a question about "car repair" is relevant to a document about "vehicle maintenance" — even though the words are different.

### Attention and Softmax — *How does a model focus on what matters?*

Not every word in a sentence is equally important when understanding another word. When you read "The bank by the river was steep", you use context to figure out "bank" means a riverbank, not a financial institution.

Attention is the mechanism that lets a model do the same thing — dynamically decide which parts of the input to focus on when processing each word. Softmax is the mathematical function that turns raw scores into probabilities, and it shows up everywhere in this process. Together, they are at the heart of how modern language models work.

### Transformer Internals — *How do the pieces assemble into a model?*

Attention does not work alone. A transformer model is built by stacking several components together — positional encodings that tell the model where each word appears in a sequence, layer normalization that keeps training stable, feed-forward networks that process information after attention, and residual connections that help gradients flow during training.

This topic shows how all those pieces fit together to form the architecture that powers GPT, BERT, LLaMA, and almost every other modern language model.

### Learning Objectives — *What is the model actually trained to do?*

A model's behavior is shaped entirely by what it was trained to predict. GPT-style models are trained to predict the next word. Embedding models are trained using contrastive objectives — pulling similar sentences closer and pushing dissimilar ones apart.

Understanding the training objective explains a lot about why models behave the way they do. It also explains why the same base architecture can produce a general chatbot, a code assistant, or a specialized embedding model depending on how it was trained.

### Training Mechanics — *How does the training process work reliably?*

Training a neural network from scratch is unstable without the right techniques. Optimizers like Adam control how weights are updated. Learning rate schedules determine how aggressively the model learns at different stages. Without these, training either diverges or converges to a poor solution.

This topic covers the practical tools that make training work — and why they matter even when you are fine-tuning a pre-trained model rather than training from scratch.

---

## How to Use This Section

Go through the topics in order if you are new to this. Each one assumes you have seen the previous.

If you already have some background, jump to the topic where your understanding starts to feel fuzzy. The goal is not to memorize formulas — it is to build an intuition for what is happening so you can reason about it when building real systems.
