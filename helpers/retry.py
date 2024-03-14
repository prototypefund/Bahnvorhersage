from functools import wraps


def retry(max_retries):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries_left = max_retries
            while True:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    retries_left -= 1
                    if retries_left == 0:
                        raise e

        return wrapper

    return decorator
