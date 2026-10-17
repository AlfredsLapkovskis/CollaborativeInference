import os
import json


class Config:
    resource_dir: str
    temp_dir: str
    experiment_dir: str

    @property
    def dnns_dir(self) -> str:
        return os.path.join(self.resource_dir, "dnns")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.resource_dir, "models")

    @property
    def data_dir(self) -> str:
        return os.path.join(self.resource_dir, "data")

    def __init__(self):
        config = {}

        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.load(f)

        self.resource_dir = config.get("resource_dir", "resources")
        self.temp_dir = config.get("temp_dir", "tmp")
        self.experiment_dir = config.get("experiment_dir", "experiments")
