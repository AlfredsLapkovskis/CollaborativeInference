import os
import glob
import argparse
from itertools import product

from utils.available_resources_model import AvailableResourcesModel
from utils.resource_consumption_model import ResourceConsumptionModel
from utils.task_completion_model import TaskCompletionModel
from utils.vector_plr_simulator import VectorPLRSimulator


def run(args):
    src_dir = args.src_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    available_resources = ["bandwidth_in",
                           "bandwidth_out", "cpu", "memory", "storage"]
    for res in available_resources:
        paths = glob.glob(os.path.join(src_dir, f"available_{res}*"))
        models = [AvailableResourcesModel.load(path) for path in paths]
        best_model = min(models, key=lambda m: m.model_aic)
        best_model.save(os.path.join(
            output_dir, f"available_{res}_model.json"), pretty=True)
        print(
            f"Best available {res} model: order={best_model.model_order}, AIC={best_model.model_aic}")

    x_download_resources = ["res_net_read",
                            "res_cpu", "res_memory", "res_storage"]
    for res in x_download_resources:
        paths = glob.glob(os.path.join(
            src_dir, f"consumption_x_download_{res}*"))
        models = [ResourceConsumptionModel.load(path) for path in paths]
        best_model = min(models, key=lambda m: m.model_aic)
        best_model.save(os.path.join(
            output_dir, f"consumption_x_download_{res}_model.json"), pretty=True)
        print(
            f"Best consumption x download {res} model: order={best_model.model_order}, AIC={best_model.model_aic}")

    y_download_resources = ["res_net_read", "res_cpu", "res_memory"]
    for res in y_download_resources:
        paths = glob.glob(os.path.join(
            src_dir, f"consumption_y_download_{res}*"))
        models = [ResourceConsumptionModel.load(path) for path in paths]
        best_model = min(models, key=lambda m: m.model_aic)
        best_model.save(os.path.join(
            output_dir, f"consumption_y_download_{res}_model.json"), pretty=True)
        print(
            f"Best consumption y download {res} model: order={best_model.model_order}, AIC={best_model.model_aic}")

    y_execution_resources = ["res_cpu", "res_memory"]
    for res in y_execution_resources:
        paths = glob.glob(os.path.join(
            src_dir, f"consumption_y_execution_{res}*"))
        models = [ResourceConsumptionModel.load(path) for path in paths]
        best_model = min(models, key=lambda m: m.model_aic)
        best_model.save(os.path.join(
            output_dir, f"consumption_y_execution_{res}_model.json"), pretty=True)
        print(
            f"Best consumption y execution {res} model: order={best_model.model_order}, AIC={best_model.model_aic}")

    y_upload_resources = ["res_net_write", "res_cpu", "res_memory"]
    for res in y_upload_resources:
        paths = glob.glob(os.path.join(
            src_dir, f"consumption_y_upload_{res}*"))
        models = [ResourceConsumptionModel.load(path) for path in paths]
        best_model = min(models, key=lambda m: m.model_aic)
        best_model.save(os.path.join(
            output_dir, f"consumption_y_upload_{res}_model.json"), pretty=True)
        print(
            f"Best consumption y upload {res} model: order={best_model.model_order}, AIC={best_model.model_aic}")

    y_download_compl_paths = glob.glob(os.path.join(
        src_dir, f"task_completion_download_model*"))
    y_execution_compl_paths = glob.glob(os.path.join(
        src_dir, f"task_completion_execution_model*"))
    y_upload_compl_paths = glob.glob(os.path.join(
        src_dir, f"task_completion_upload_model*"))
    y_download_compl_models = [TaskCompletionModel.load(
        path) for path in y_download_compl_paths]
    y_execution_compl_models = [TaskCompletionModel.load(
        path) for path in y_execution_compl_paths]
    y_upload_compl_models = [TaskCompletionModel.load(
        path) for path in y_upload_compl_paths]

    y_task_completion_models = []

    for download_model, execution_model, upload_model in product(y_download_compl_models, y_execution_compl_models, y_upload_compl_models):
        try:
            _ = VectorPLRSimulator(
                task_completion_models=[download_model,
                                        execution_model, upload_model],
            )

            y_task_completion_models.append(
                (download_model, execution_model, upload_model))
        except AssertionError as e:
            # Model assumptions not satisfied, skip this combination
            pass

    best_y_download_compl_model, best_y_execution_compl_model, best_y_upload_compl_model = min(
        y_task_completion_models,
        key=lambda models: sum(m.model_aic for m in models) / 3,
    )

    for state, best_model in zip(["download", "execution", "upload"], [best_y_download_compl_model, best_y_execution_compl_model, best_y_upload_compl_model]):
        best_model.save(os.path.join(
            output_dir, f"task_completion_{state}_model.json"), pretty=True)
        print(
            f"Best task completion {state} model: AIC={best_model.model_aic}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True,
                        help="Directory containing the model files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the best model files")

    args = parser.parse_args()
    run(args)
