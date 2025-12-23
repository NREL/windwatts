import random

messages = [
    "Hello, World!",
    "Goodbye, World!",
    "I am a message!",
    "This is a message!",
    "Random message!",
]


def random_message():
    return random.choice(messages)
