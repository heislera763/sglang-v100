"""Read the existing API key from the environment, keeping it out of argv."""

import os


def arguments(original, *args, **kwargs):
    result = original(*args, **kwargs)
    if result.api_key is None:
        result.api_key = os.environ.get("LLAMA_API_KEY")
    return result


def redact_logs():
    import logging

    secret = os.environ.get("LLAMA_API_KEY")
    if not secret:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        message = record.getMessage()
        if secret in message:
            record.msg = message.replace(secret, "[REDACTED]")
            record.args = ()
        return record

    logging.setLogRecordFactory(factory)
