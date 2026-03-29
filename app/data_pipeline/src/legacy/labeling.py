# labeling.py

def label_attack_span(adv_text, original_prompt):
    span = adv_text.replace(original_prompt, "").strip()

    # crude token split
    tokens = span.split()

    return tokens