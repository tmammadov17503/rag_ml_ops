DATA_DIR = "/app/data"


def _read_corpus(data_dir: str):
    import os

    texts = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith((".txt", ".md")):
                fp = os.path.join(root, name)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        t = f.read().strip()
                        if t:
                            texts.append(t)
                except Exception:
                    pass
    return texts
