import random
import copy

import datetime

import json
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_file_path, "adjectives.json"), "r") as f:
    adjectives = json.load(f)

adjectives = [str(adj).capitalize() for adj in adjectives]

second_adjectives = copy.deepcopy(adjectives)

with open(os.path.join(current_file_path, "nouns.json"), "r") as f:
    nouns = json.load(f)


def generate_name():
    seconds = datetime.datetime.now().second
    return (
        second_adjectives[int(seconds) % len(second_adjectives)]
        + "-"
        + random.choice(adjectives)
        + "-"
        + random.choice(nouns)
    )

