# augment.py
import random

def obfuscate_text(text):
    replacements = {
        "ignore": "1gn0r3",
        "instructions": "1nstruct10ns",
        "previous": "pr3v10us"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text

def paraphrase_simple(text):
    variations = [
        text,
        text.replace("and", "then"),
        text.replace("please", "kindly"),
    ]
    return random.choice(variations)