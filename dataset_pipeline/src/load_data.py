# load_data.py
import random

def load_benign_data():
    base_prompts = [
        "Summarize this document",
        "Explain neural networks",
        "What is gradient descent?",
        "Translate this sentence to French",
        "Write a short story about space"
    ]

    expanded = []
    for p in base_prompts:
        expanded.append(p)
        expanded.append(f"Can you {p.lower()}?")
        expanded.append(f"Please {p.lower()}")

    return expanded