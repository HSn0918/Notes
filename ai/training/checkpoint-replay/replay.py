"""Compare an uninterrupted scalar regression with two restart strategies."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

SAMPLES = ((1., 3.), (2., 5.), (3., 7.), (4., 9.))
RATE = 0.02
MOMENTUM = 0.8
TOTAL_STEPS = 12


@dataclass
class State:
    weight: float = 0.
    velocity: float = 0.
    step: int = 0
    cursor: int = 0


def train(state, steps):
    seen = []
    for _ in range(steps):
        index = state.cursor % len(SAMPLES)
        x, y = SAMPLES[index]
        gradient = (state.weight * x - y) * x
        velocity = MOMENTUM * state.velocity + gradient
        state = State(state.weight - RATE * velocity, velocity, state.step + 1, state.cursor + 1)
        seen.append(index)
    return state, seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut", type=int, default=5, choices=range(1, TOTAL_STEPS))
    args = parser.parse_args()
    stop = args.cut
    baseline, order = train(State(), TOTAL_STEPS)
    saved, prefix = train(State(), stop)
    with TemporaryDirectory(prefix="notes-checkpoint-") as tmp:
        path = Path(tmp) / "state.json"
        path.write_text(json.dumps(asdict(saved)))
        restored = State(**json.loads(path.read_text()))
    for label, state in (("full", restored), ("omit_cursor", replace(restored, cursor=0)),
                         ("omit_momentum", replace(restored, velocity=0.))):
        end, suffix = train(state, TOTAL_STEPS - stop)
        same_order = prefix + suffix == order
        print(f"{label}: next_sample={suffix[0]} same_order={same_order} weight_error={abs(end.weight-baseline.weight):.6g}")
        if label == "full" and (end != baseline or not same_order):
            raise AssertionError("complete checkpoint did not reproduce the baseline")


if __name__ == "__main__":
    main()
