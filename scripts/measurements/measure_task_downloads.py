import os
import csv
from multiprocessing import Process
import websockets
import asyncio
import aiofiles
from time import sleep
from argparse import ArgumentParser
from time import time
from datetime import datetime

from utils.resource_monitor import ResourceMonitor
from utils.io_logger import IOLogger, IOEvent
from utils.rate_limited_client_connection import RateLimitedClientConnection
from utils.datetime import now_str
from models.dnns import *
from models.experiments import MeasurementExperiment
from config import Config


HOST = "127.0.0.1"
PORT = 8888

CHUNK_SIZE = 1024 * 1024
MAX_MESSAGE_SIZE = CHUNK_SIZE << 1

models = get_all_pytorch_models()
model_names = [model.get_name() for model in models]
model_names_set = set(model_names)
config = Config()


def client_process(bandwidth, model_name, io_logger: IOLogger):
    asyncio.run(run_client(bandwidth, model_name, io_logger))


async def run_client(bandwidth, model_name, io_logger: IOLogger):
    if not os.path.exists(config.temp_dir):
        os.makedirs(config.temp_dir, exist_ok=True)

    create_connection = RateLimitedClientConnection.create_factory(
        send_rate_limit=None,
        recv_rate_limit=bandwidth,
    )

    async with websockets.connect(f"ws://{HOST}:{PORT}",
                                  max_size=MAX_MESSAGE_SIZE,
                                  compression=None,
                                  create_connection=create_connection,
                                  ping_interval=None,
                                  ) as websocket:
        await websocket.send(model_name)
        io_logger.log(IOEvent.NET_WRITE, len(model_name), time())

        save_path = get_model_path(config, model_name)

        async def write_data(data):
            async with aiofiles.open(save_path, 'ab') as f:
                await f.write(data)
                io_logger.log(IOEvent.DISK_WRITE, len(data), time())

        async with asyncio.TaskGroup() as tg:
            while True:
                response = await websocket.recv()
                io_logger.log(IOEvent.NET_READ,
                              len(response), time())

                name_len = response[0]
                model_name = response[1: name_len + 1].decode()
                model_data = response[name_len + 1:]

                if model_data:
                    tg.create_task(write_data(model_data))
                else:
                    break

        os.remove(save_path)
        await websocket.close()


def server_process():
    asyncio.run(run_server())


async def run_server():
    server = None
    try:
        server = await websockets.serve(lambda conn: server_handler(conn, server), HOST, PORT, max_size=MAX_MESSAGE_SIZE, compression=None, ping_interval=None)
        await server.wait_closed()
    except Exception as exc:
        print(f"Server error: {exc}")


async def server_handler(connection: websockets.ServerConnection, server: websockets.Server):
    async for message in connection:
        if message in model_names_set:
            name_len = len(message).to_bytes(1)
            model_path = os.path.join(config.models_dir, f"{message}.pth")

            async with aiofiles.open(model_path, 'rb') as f:
                while chunk := await f.read(CHUNK_SIZE):
                    out_message = name_len + message.encode() + chunk
                    await connection.send(out_message)
            await connection.send(name_len + message.encode())
        elif message == "CLOSE":
            server.close()


def get_model_path(config: Config, model_name):
    return os.path.join(config.temp_dir, f"{model_name}.pth")


def get_stored_model_size(config, model_name):
    path = get_model_path(config, model_name)
    return os.path.getsize(path) if os.path.exists(path) else 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str,
                        required=True, help="Experiment handle")
    args = parser.parse_args()

    experiment = MeasurementExperiment(args.experiment)

    server_p = Process(target=server_process)
    server_p.start()

    sleep(5)  # Wait for the server to start

    log_file = os.path.join(
        config.temp_dir, f"task_downloads_exp_{experiment.experiment_handle}_{now_str()}.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    header_written = False

    is_interrupted = False

    for bandwidth_in in experiment.net_in_limits:
        bandwidth_in //= 8  # to bytes
        if is_interrupted:
            break
        for trial in range(experiment.trials):
            if is_interrupted:
                break
            for model_name in model_names:
                print(f"BEGIN {bandwidth_in=}, {trial=}, {model_name=}.")

                io_logger = IOLogger()

                client_p = Process(target=client_process, args=(
                    bandwidth_in, model_name, io_logger))
                client_p.start()

                monitor = ResourceMonitor(
                    pid=client_p.pid, io_logger=io_logger)
                initial_log_written = False

                while client_p.is_alive():
                    try:
                        monitor.start()
                        if initial_log_written:
                            # Wait for exchanging approx. 3 chunks
                            sleep(CHUNK_SIZE / bandwidth_in * 3)
                        initial_log_written = True
                        usage = monitor.stop()

                        record = {
                            "conf_trial": trial,
                            "conf_bandwidth_in": bandwidth_in,
                            "conf_model_name": model_name,
                            "res_cpu": usage.cpu,
                            "res_memory": usage.memory,
                            "res_net_read": usage.net_read,
                            "res_net_write": usage.net_write,
                            "res_disk_read": usage.disk_read,
                            "res_disk_write": usage.disk_write,
                            "res_storage": get_stored_model_size(config, model_name),
                            "timestamp": f"{now_str()}"
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
