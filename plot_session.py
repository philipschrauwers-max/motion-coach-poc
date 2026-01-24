import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def load_session(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    reps = data.get("reps", [])
    return data, reps


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_session.py data/sessions/session-YYYYMMDD-HHMMSS.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    data, reps = load_session(path)

    if not reps:
        print("No reps in session.")
        return

    # Time axis relative to first rep
    t0 = reps[0]["t"]
    times = [r["t"] - t0 for r in reps]
    scores = [r["score"] for r in reps]
    kinds = [r["kind"] for r in reps]

    # Plot 1: score over time (all reps)
    plt.figure()
    plt.title("Scores over time (all reps)")
    plt.xlabel("Seconds")
    plt.ylabel("Score")
    plt.plot(times, scores, marker="o", linestyle="-")
    plt.grid(True)
    plt.show()

    # Plot 2: per-technique trend
    by_kind = {}
    for r in reps:
        by_kind.setdefault(r["kind"], []).append(r)

    plt.figure()
    plt.title("Scores over time (by technique)")
    plt.xlabel("Seconds")
    plt.ylabel("Score")
    for k, rs in by_kind.items():
        t0k = rs[0]["t"]
        tk = [x["t"] - t0 for x in rs]  # keep same baseline as all reps
        sk = [x["score"] for x in rs]
        plt.plot(tk, sk, marker="o", linestyle="-", label=k)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
