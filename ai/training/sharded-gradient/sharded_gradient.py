"""Check a softmax training step against finite differences and uneven shards."""

import argparse
from math import exp, log


SAMPLES = [((2, 1), 0), ((-1, 2), 1), ((1, -2), 0), ((-2, 1), 1)]
WEIGHTS = [[0.1, -0.2], [0.05, 0.3]]
DIFF_STEP = 1e-5
DIFF_TOL = 1e-8
MERGE_TOL = 1e-12
LEARNING_RATE = 0.1


def loss_and_grad(samples, weights):
    if not samples:
        raise ValueError("mean loss requires at least one valid token")
    grad = [[0.0 for _ in row] for row in weights]
    total = 0.0
    for x, label in samples:
        logits = [sum(v * w for v, w in zip(x, col)) for col in zip(*weights)]
        peak = max(logits)
        exps = [exp(z - peak) for z in logits]
        denom = sum(exps)
        total += log(denom) - (logits[label] - peak)
        for i, value in enumerate(x):
            for j, numerator in enumerate(exps):
                grad[i][j] += value * (numerator / denom - (j == label))
    return total / len(samples), [[g / len(samples) for g in row] for row in grad]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-tokens", type=int, choices=range(1, len(SAMPLES)), default=1)
    cut = parser.parse_args().left_tokens
    before, grad = loss_and_grad(SAMPLES, WEIGHTS)
    left, right = SAMPLES[:cut], SAMPLES[cut:]
    _, a = loss_and_grad(left, WEIGHTS)
    _, b = loss_and_grad(right, WEIGHTS)
    diff_error = merge_error = naive_error = 0.0
    for i, row in enumerate(grad):
        for j, expected in enumerate(row):
            plus, minus = [r[:] for r in WEIGHTS], [r[:] for r in WEIGHTS]
            plus[i][j] += DIFF_STEP
            minus[i][j] -= DIFF_STEP
            numerical = (loss_and_grad(SAMPLES, plus)[0] - loss_and_grad(SAMPLES, minus)[0]) / (2 * DIFF_STEP)
            weighted = (len(left) * a[i][j] + len(right) * b[i][j]) / len(SAMPLES)
            diff_error = max(diff_error, abs(numerical - expected))
            merge_error = max(merge_error, abs(weighted - expected))
            naive_error = max(naive_error, abs((a[i][j] + b[i][j]) / 2 - expected))

    updated = [[w - LEARNING_RATE * g for w, g in zip(ws, gs)] for ws, gs in zip(WEIGHTS, grad)]
    after, _ = loss_and_grad(SAMPLES, updated)
    print(f"finite_diff_error={diff_error:.3e} weighted_error={merge_error:.3e} naive_error={naive_error:.3e}")
    print(f"train_loss={before:.6f} -> {after:.6f}")
    equal_sizes = len(left) == len(right)
    print(f"tokens={len(left)}:{len(right)}")
    naive_matches = naive_error < MERGE_TOL
    if not (diff_error < DIFF_TOL and merge_error < MERGE_TOL and naive_matches == equal_sizes and after < before):
        raise AssertionError("gradient check, uneven-shard control or SGD step failed")
    print("SELF-CHECK: PASS")


if __name__ == "__main__":
    main()
