import os
import csv
import io
import numpy as np
from multiprocessing import Process
import websockets
import asyncio
import aiofiles
from time import sleep
from argparse import ArgumentParser
from time import time
from datetime import datetime
from random import Random
import struct
from PIL import Image

from utils.resource_monitor import ResourceMonitor
from utils.io_logger import IOLogger, IOEvent
from utils.rate_limited_client_connection import RateLimitedClientConnection
from utils.datetime import now_str
from models.dnns import *
from models.experiments import MeasurementExperiment
from config import Config


HOST = "127.0.0.1"
PORT = 8889

CHUNK_SIZE = 1024 * 1024
MAX_MESSAGE_SIZE = CHUNK_SIZE << 1

models = get_all_pytorch_models()
model_names = [model.get_name() for model in models]
model_names_set = set(model_names)
config = Config()


def client_process(bandwidth_in, bandwidth_out, model: DNN, n_tasks, io_logger: IOLogger):
    asyncio.run(run_client(bandwidth_in, bandwidth_out,
                model, n_tasks, io_logger))


async def run_client(bandwidth_in, bandwidth_out, model: DNN, n_tasks, io_logger: IOLogger):
    assert n_tasks > 0

    create_connection = RateLimitedClientConnection.create_factory(
        send_rate_limit=bandwidth_out,
        recv_rate_limit=bandwidth_in,
    )

    async with websockets.connect(f"ws://{HOST}:{PORT}",
                                  max_size=MAX_MESSAGE_SIZE,
                                  compression=None,
                                  create_connection=create_connection,
                                  #   ping_interval=None,
                                  ) as websocket:
        request_message = f"IMAGES_{n_tasks}"
        await websocket.send(request_message)

        image_data_batch = []
        for _ in range(n_tasks):
            image_data = bytearray()

            while True:
                response = await websocket.recv()
                io_logger.log(IOEvent.NET_READ, len(response), time())

                progress_token = response[0]
                data = response[1:]

                if progress_token and data:
                    image_data.extend(data)
                else:
                    break
            image_data_batch.append(image_data)

        model_path = os.path.join(config.models_dir, f"{model.get_name()}.pth")
        assert os.path.exists(model_path)

        model.load_from_file(model_path)

        def generate_images():
            for image_data in image_data_batch:
                with Image.open(io.BytesIO(image_data)) as img:
                    yield img.convert("RGB")

        proba = model.predict_proba_batch(generate_images())

        proba_count = len(proba[0]).to_bytes(2)
        proba_bytes = struct.pack(
            f"{len(proba[0]) * n_tasks}f", *np.reshape(proba, -1))
        proba_message = proba_count + proba_bytes

        io_logger.log(IOEvent.NET_WRITE, 0, time(), wait=True, force=True)
        await websocket.send(proba_message)
        io_logger.log(IOEvent.NET_WRITE, len(proba_message),
                      time(), wait=True, force=True)

        print(f"Inference finished.")
        await websocket.close()
        # Simple workaround to digest the latest net write logs.
        sleep(1)


def server_process(image_dir):
    asyncio.run(run_server(image_dir))


async def run_server(image_dir):
    server = None
    server = await websockets.serve(lambda conn: server_handler(conn, server, image_dir), HOST, PORT, max_size=MAX_MESSAGE_SIZE, compression=None)
    try:
        await server.wait_closed()
    except Exception as exc:
        print(f"Server error: {exc}")


async def server_handler(connection: websockets.ServerConnection, server: websockets.Server, image_dir):
    random = Random()
    image_names = os.listdir(image_dir)
    assert image_names

    async for message in connection:
        if message.startswith("IMAGES"):
            image_count = int(message.split("_")[1])
            image_names = random.choices(image_names, k=image_count)

            for image_name in image_names:
                image_path = os.path.join(image_dir, image_name)

                in_progress_token = (1).to_bytes(1)
                finished_token = (0).to_bytes(1)

                async with aiofiles.open(image_path, 'rb') as f:
                    while chunk := await f.read(CHUNK_SIZE):
                        out_message = in_progress_token + chunk
                        await connection.send(out_message)
                await connection.send(finished_token)

            inference_proba_bytes = await connection.recv()
            inference_proba_count = int.from_bytes(inference_proba_bytes[:2])
            inference_proba_bytes = inference_proba_bytes[2:]

            inference_proba = struct.unpack(
                f"{inference_proba_count * image_count}f", inference_proba_bytes)
            inference_proba = np.reshape(
                inference_proba, (-1, inference_proba_count))
            print(f"Received inference probabilities: {inference_proba.shape}")
        elif message == "CLOSE":
            server.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str,
                        required=True, help="Experiment handle")
    parser.add_argument("--image_dir", type=str,
                        required=True, help="Directory with images")
    args = parser.parse_args()

    experiment = MeasurementExperiment(args.experiment)

    server_p = Process(target=server_process, args=(args.image_dir,))
    server_p.start()

    sleep(5)  # Wait for the server to start

    log_file = os.path.join(
        config.temp_dir, f"task_executions_exp_{experiment.experiment_handle}_{now_str()}.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    header_written = False

    is_interrupted = False
    for bandwidth_in in experiment.net_in_limits:
        bandwidth_in //= 8  # to bytes
        for bandwidth_out in experiment.net_out_limits:
            bandwidth_out //= 8  # to bytes
            for trial in range(experiment.trials):
                for task_count in experiment.task_counts:
                    if is_interrupted:
                        break
                    for model in models:
                        print(
                            f"BEGIN {trial=}, {bandwidth_in=}, {bandwidth_out=}, {task_count=}, model_name={model.get_name()}.")

                        io_logger = IOLogger()

                        client_p = Process(target=client_process, args=(
                            bandwidth_in, bandwidth_out, model, task_count, io_logger))
                        client_p.start()

                        monitor = ResourceMonitor(
                            pid=client_p.pid, io_logger=io_logger)
                        initial_log_written = False

                        while client_p.is_alive():
                            try:
                                monitor.start()
                                if initial_log_written:
                                    sleep(0.1)
                                initial_log_written = True
                                usage = monitor.stop()

                                record = {
                                    "conf_trial": trial,
                                    "conf_bandwidth_in": bandwidth_in,
                                    "conf_bandwidth_out": bandwidth_out,
                                    "conf_task_count": task_count,
                                    "conf_model_name": model.get_name(),
                                    "res_cpu": usage.cpu,
                                    "res_memory": usage.memory,
                                    "res_net_read": usage.net_read,
                                    "res_net_write": usage.net_write,
                                    "res_disk_read": usage.disk_read,
                                    "res_disk_write": usage.disk_write,
                                    "timestamp": now_str()
                                }

                                with open(log_file, "a") as f:
                                    writer = csv.DictWriter(
                                        f, fieldnames=record.keys())
                                    if not header_written:
                                        writer.writeheader()
                                        header_written = True
                                    writer.writerow(record)
                            except KeyboardInterrupt as exc:
                                is_interrupted = True
                                break
                            except BaseException as exc:
                                print(f"Exception: {exc}")

                        io_logger.close(wait=True)
                        monitor.close(wait=True)
                        client_p.terminate()
                        client_p.join()

                        if is_interrupted:
                            break

    server_p.terminate()
    server_p.join()
    print("SERVER JOINED")
