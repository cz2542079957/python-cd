import time
from functools import wraps


def run_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"执行时间：{elapsed_time} 秒\n")
        return result

    return wrapper
