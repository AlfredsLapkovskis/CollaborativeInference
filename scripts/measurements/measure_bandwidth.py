import os
import csv
import speedtest
from datetime import datetime
from time import sleep

from config import Config

if __name__ == "__main__":
    config = Config()
    speed = speedtest.Speedtest()

    log_file = os.path.join(config.temp_dir, f"bandwidth_{now_str()}.csv")
    os.makedirs(config.temp_dir, exist_ok=True)

    try:
        i = 0
        header_written = False

        while True:
            print(f"Measurement {i}")
            speed.download()
            speed.upload()
            results_dict = speed.results.dict()
            results_dict["timestamp"] = f"{now_str()}"
            print(results_dict)

            with open(log_file, "a") as f:
                writer = csv.DictWriter(f, fieldnames=results_dict.keys())
                if not header_written:
                    writer.writeheader()
                    header_written = True
                writer.writerow(results_dict)
            sleep(1)
            i += 1
    except KeyboardInterrupt:
        print("Finished")
