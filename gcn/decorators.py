import functools


def track_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Starting execution of {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"Finished execution of {func.__name__}.")
        return result

    return wrapper
