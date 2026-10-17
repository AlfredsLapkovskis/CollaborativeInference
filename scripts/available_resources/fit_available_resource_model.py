import os
import argparse
import pandas as pd
from datetime import datetime

from models.experiments import AvailableResourceModelExperiment
from utils.available_resources_model import *
from utils.datetime import now_str


def run(experiment):
    df_path = experiment.df_path
    series_name = experiment.series_name
    arima_order = experiment.arima_order
    output_dir = experiment.output_dir

    df = pd.read_csv(df_path)
    X_series = df[series_name].to_numpy()

    trainer = AvailableResourcesModelTrainer(
        series=X_series,
        order=arima_order,
    )
    config = trainer.fit()

    order_str = '-'.join(str(o) for o in arima_order)
    save_path = os.path.join(
        output_dir, f"available_{series_name}_model_{order_str}_{now_str()}.json")

    config.save(save_path, pretty=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment handle")

    args = parser.parse_args()
    experiment = AvailableResourceModelExperiment(args.experiment)
    run(experiment)
