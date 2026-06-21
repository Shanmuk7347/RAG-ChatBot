from tenacity import retry, stop_after_attempt, before_sleep_log, wait_exponential
from logger import logger

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10),
    before_sleep=lambda k: logger.warning(f"Retrying for {k.fn.__name__} ({k.attempt_number}/3) because {type(k.outcome.exception()).__name__}"), reraise=True)
def safe_invoke(runnable, payload):
    return runnable.invoke(payload)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10),
    before_sleep=lambda k: logger.warning(f"Retrying for {k.fn.__name__} ({k.attempt_number}/3) because {type(k.outcome.exception()).__name__}"), reraise=True)
def safe_stream(runnable, payload):
    return runnable.stream(payload)