import os
import csv
import psutil
import shutil
from utils.datetime import now_str

from config import Config


if __name__ == "__main__":
    config = Config()
    log_file = os.path.join(config.temp_dir, f"resources_{now_str()}.csv")
    header_written = False

    try:
        while True:
            storage_total, storage_used, storage_free = shutil.disk_usage(
                os.path.sep)

            memory = psutil.virtual_memory()
            memory_total = memory.total
            memory_available = memory.available
            memory_used = memory.used
            memory_percent = memory.percent  # used

            cpu_percent = sum(psutil.cpu_percent(
                interval=3, percpu=True))  # used
            cpu_total = psutil.cpu_count() * 100.

            record = dict(
                storage_total=storage_total,
                storage_used=storage_used,
                storage_free=storage_free,
                memory_total=memory_total,
                memory_available=memory_available,
                memory_used=memory_used,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                cpu_total=cpu_total,
                timestamp=f"{now_str()}",
            )

            with open(log_file, "a") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                if not header_written:
                    writer.writeheader()
                    header_written = True
                writer.writerow(record)
    except KeyboardInterrupt:
        pass
