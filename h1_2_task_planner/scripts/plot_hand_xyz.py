#!/usr/bin/env python3

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_axis(df, output_dir, condition, hand, axis):
    subset = df[
        (df["condition"] == condition) &
        (df["hand"] == hand)
    ].copy()

    if subset.empty:
        return

    plt.figure(figsize=(8, 4.5))

    for trial, trial_df in subset.groupby("trial"):
        trial_df = trial_df.sort_values("time_s")
        plt.plot(
	    trial_df["time_s"].to_numpy(),
            trial_df[f"{axis}_m"].to_numpy(),
            linewidth=1.4,
            label=f"Ensayo {trial}"
        )

    plt.xlabel("Tiempo [s]")
    plt.ylabel(f"{axis.upper()} de la mano [m]")
    plt.title(f"{hand} - {condition} - eje {axis.upper()}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    safe_condition = condition.replace("/", "_").replace(" ", "_")
    safe_hand = hand.replace("/", "_")

    out_path = os.path.join(
        output_dir,
        f"hand_xyz_{safe_hand}_{safe_condition}_{axis}.png"
    )

    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/tmp/h1_2_results/actual_hand_poses.csv",
        help="CSV generado por hand_pose_trial_logger.py"
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/h1_2_results/hand_xyz_plots",
        help="Carpeta de salida para las figuras"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)

    required_columns = {
        "condition", "trial", "time_s", "hand", "x_m", "y_m", "z_m"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas en el CSV: {missing}")

    for condition in sorted(df["condition"].unique()):
        for hand in sorted(df["hand"].unique()):
            for axis in ["x", "y", "z"]:
                plot_axis(df, args.output_dir, condition, hand, axis)

    print(f"Figuras guardadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
