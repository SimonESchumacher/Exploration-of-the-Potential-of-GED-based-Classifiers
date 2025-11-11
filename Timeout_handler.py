import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    # This function is called when the signal is triggered
    raise TimeoutException("Function call timed out!")

def run_with_timeout(func, args=(), timeout_seconds=60):
    # Set the signal handler and the alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        # Call the function
        result = func(*args)
    finally:
        # Disable the alarm regardless of success or failure
        signal.alarm(0)
    return result

# Example usage (assuming your training function is my_svc_fit)
# result = run_with_timeout(my_svc_fit, args=(X_train, y_train), timeout_seconds=300) # 5 minutes