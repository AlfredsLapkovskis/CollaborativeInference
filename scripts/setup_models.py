import os

from config import Config
from models.dnns import *


if __name__ == "__main__":
    config = Config()

    models = get_all_pytorch_models()

    if not os.path.exists(config.dnns_dir):
        os.makedirs(config.dnns_dir, exist_ok=True)

    for model in models:
        model.load_from_hub()
        print(f"Loaded {model.get_name()} model from hub")

    for model in models:
        path = os.path.join(config.dnns_dir, f"{model.get_name()}.pth")
        model.save(path)
        print(f"Saved {model.get_name()} model to {path}")
