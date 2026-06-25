#!/usr/bin/env python3
"""Generate loss_curve.png from r16 trainer_state.json (Colab export)."""

import json
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).parent.resolve()
TRAINER_STATE = ROOT / "lab21_lora_t4/r16/checkpoint-69/trainer_state.json"
OUT = ROOT / "lab21_lora_t4/loss_curve.png"


def main():
    with open(TRAINER_STATE) as f:
        state = json.load(f)

    df = pd.DataFrame(state["log_history"])
    train = df[df["loss"].notna()]

    plt.figure(figsize=(8, 4))
    plt.plot(train["step"], train["loss"], label="train", color="#0E2A52", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve — r=16")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {OUT}")
    print(f"  Steps: {len(train)}, loss {train['loss'].iloc[0]:.3f} → {train['loss'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
