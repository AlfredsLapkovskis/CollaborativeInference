from datetime import datetime


def now_str():
    """Get the current date and time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
