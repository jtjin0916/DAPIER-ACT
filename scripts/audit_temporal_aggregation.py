#!/usr/bin/env python3

import numpy as np


def weights_current_reference(n, k=0.01):
    """Current reference: history iteration is oldest -> newest."""
    w = np.exp(-k * np.arange(n, dtype=np.float32))
    return w / w.sum()


def weights_newest_heavy(n, k=0.01):
    """Candidate corrected weighting for a future A/B experiment."""
    w = np.exp(-k * np.arange(n - 1, -1, -1, dtype=np.float32))
    return w / w.sum()


def main():
    for n in (2, 3, 5, 20):
        current = weights_current_reference(n)
        corrected = weights_newest_heavy(n)

        print(f"N={n}")
        print("history order: oldest -> newest")
        print("current:  ", np.round(current, 6).tolist())
        print("candidate:", np.round(corrected, 6).tolist())
        print(
            "current max:",
            "oldest" if int(np.argmax(current)) == 0 else "newest",
        )
        print(
            "candidate max:",
            "oldest" if int(np.argmax(corrected)) == 0 else "newest",
        )
        print()

    print(
        "NOTE: the verified reference comments say newer predictions should "
        "receive larger weights, but the current array ordering gives the "
        "largest weight to the oldest prediction. Keep the reference unchanged "
        "and evaluate this as an explicit A/B experiment."
    )


if __name__ == "__main__":
    main()
