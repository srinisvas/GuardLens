import json
import pandas as pd

def load_kaggle_dataset(path):
    df = pd.read_csv(path)

    data = []
    for i, row in df.iterrows():
        text = row["prompt"]
        label = 1 if row["label"] == "injection" else 0

        data.append({
            "id": f"k_{i}",
            "text": text,
            "label": label,
            "attack_type": "real_world"
        })
    return data


def load_jailbreak_dataset(path):
    with open(path, "r") as f:
        raw = json.load(f)

    data = []
    for i, item in enumerate(raw):
        text = item.get("prompt", "")
        data.append({
            "id": f"j_{i}",
            "text": text,
            "label": 1,
            "attack_type": "jailbreak"
        })
    return data