"""Mapping functions"""


def to_list(message):
    """Puts message into a list"""
    if message is None:
        return []
    if isinstance(message, list):
        return message

    return [message]
