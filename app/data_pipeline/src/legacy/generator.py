# generator.py
import random
from templates import ATTACK_TEMPLATES


def generate_adversarial(prompt):
    attack_type = random.choice(list(ATTACK_TEMPLATES.keys()))
    template = random.choice(ATTACK_TEMPLATES[attack_type])

    adv_prompt = template.format(prompt=prompt)

    return {
        "text": adv_prompt,
        "label": 1,
        "attack_type": attack_type,
        "original": prompt
    }


def generate_pair(prompt):
    benign = {
        "text": prompt,
        "label": 0,
        "attack_type": "none"
    }

    adversarial = generate_adversarial(prompt)

    return benign, adversarial