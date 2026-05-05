#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


CONDITION_ORDER = ["P1_cercana", "P2_intermedia", "P3_borde"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_bar(summary, metric, ylabel, title, output_path):
    data = summary[summary["success"] == 1].copy()

    if data.empty:
        print(f"No hay datos exitosos para graficar {metric}")
        return

    stats = data.groupby("condition")[metric].agg(["mean", "std"]).reindex(CONDITION_ORDER)
    stats = stats.dropna(subset=["mean"])

    plt.figure(figsize=(7, 4))
    plt.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=5)
    plt.ylabel(ylabel)
    plt.xlabel("Condición experimental")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")

def plot_joint_tracking(input_dir, output_dir, condition, trial, joint_name):
    planned_path = os.path.join(input_dir, "planned_joint_trajectory.csv")
    actual_path = os.path.join(input_dir, "actual_joint_states.csv")

    if not os.path.exists(planned_path):
        print("No existe planned_joint_trajectory.csv")
        return

    if not os.path.exists(actual_path):
        print("No existe actual_joint_states.csv")
        return

    planned = pd.read_csv(planned_path)
    actual = pd.read_csv(actual_path)

    planned_f = planned[
        (planned["condition"] == condition) &
        (planned["trial"] == trial) &
        (planned["joint_name"] == joint_name)
    ].copy()

    actual_f = actual[
        (actual["condition"] == condition) &
        (actual["trial"] == trial) &
        (actual["joint_name"] == joint_name)
    ].copy()

    if planned_f.empty or actual_f.empty:
        print(f"No hay datos suficientes para {condition}, trial {trial}, joint {joint_name}")
        return

    planned_f["time_s"] = pd.to_numeric(planned_f["time_s"], errors="coerce")
    planned_f["desired_position_rad"] = pd.to_numeric(
        planned_f["desired_position_rad"], errors="coerce"
    )

    actual_f["time_s"] = pd.to_numeric(actual_f["time_s"], errors="coerce")
    actual_f["position_rad"] = pd.to_numeric(
        actual_f["position_rad"], errors="coerce"
    )

    planned_f = planned_f.dropna(subset=["time_s", "desired_position_rad"])
    actual_f = actual_f.dropna(subset=["time_s", "position_rad"])

    planned_f = planned_f.sort_values("time_s")
    actual_f = actual_f.sort_values("time_s")

    plt.figure(figsize=(8, 4.5))

    plt.plot(
        planned_f["time_s"].to_numpy(),
        planned_f["desired_position_rad"].to_numpy(),
        label="Trayectoria planificada"
    )

    plt.plot(
        actual_f["time_s"].to_numpy(),
        actual_f["position_rad"].to_numpy(),
        label="Estado articular ejecutado",
        linestyle="--"
    )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Posición articular [rad]")
    plt.title(f"Seguimiento articular: {joint_name} ({condition}, repetición {trial})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"seguimiento_{joint_name}_{condition}_trial_{trial}.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")

def create_summary_tables(summary, output_dir):
    successful = summary[summary["success"] == 1].copy()

    if successful.empty:
        print("No hay ejecuciones exitosas para generar resumen.")
        return

    grouped = successful.groupby("condition").agg(
        repeticiones_exitosas=("success", "sum"),
        tiempo_planificacion_prom_s=("planning_time_s", "mean"),
        tiempo_planificacion_std_s=("planning_time_s", "std"),
        tiempo_ejecucion_prom_s=("execution_time_s", "mean"),
        tiempo_ejecucion_std_s=("execution_time_s", "std"),
        duracion_planificada_prom_s=("planned_duration_s", "mean"),
        puntos_trayectoria_prom=("trajectory_points", "mean"),
        error_cartesiano_prom_mm=("cartesian_error_mm", "mean"),
        error_cartesiano_std_mm=("cartesian_error_mm", "std"),
        error_articular_prom_rad=("joint_error_rad", "mean"),
        error_articular_std_rad=("joint_error_rad", "std"),
    ).reindex(CONDITION_ORDER)

    csv_path = os.path.join(output_dir, "summary_grouped_metrics.csv")
    tex_path = os.path.join(output_dir, "summary_grouped_metrics.tex")

    grouped.to_csv(csv_path)
    grouped.round(4).to_latex(tex_path)

    print(f"Guardado: {csv_path}")
    print(f"Guardado: {tex_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/tmp/h1_2_results")
    parser#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


CONDITION_ORDER = ["P1_cercana", "P2_intermedia", "P3_borde"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_bar(summary, metric, ylabel, title, output_path):
    data = summary[summary["success"] == 1].copy()

    if data.empty:
        print(f"No hay datos exitosos para graficar {metric}")
        return

    stats = data.groupby("condition")[metric].agg(["mean", "std"]).reindex(CONDITION_ORDER)
    stats = stats.dropna(subset=["mean"])

    plt.figure(figsize=(7, 4))
    plt.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=5)
    plt.ylabel(ylabel)
    plt.xlabel("Condición experimental")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")


def plot_joint_tracking(input_dir, output_dir, condition, trial, joint_name):
    planned_path = os.path.join(input_dir, "planned_joint_trajectory.csv")
    actual_path = os.path.join(input_dir, "actual_joint_states.csv")

    if not os.path.exists(planned_path):
        print("No existe planned_joint_trajectory.csv")
        return

    if not os.path.exists(actual_path):
        print("No existe actual_joint_states.csv")
        return

    planned = pd.read_csv(planned_path)
    actual = pd.read_csv(actual_path)

    planned_f = planned[
        (planned["condition"] == condition) &
        (planned["trial"] == trial) &
        (planned["joint_name"] == joint_name)
    ]

    actual_f = actual[
        (actual["condition"] == condition) &
        (actual["trial"] == trial) &
        (actual["joint_name"] == joint_name)
    ]

    if planned_f.empty or actual_f.empty:
        print(f"No hay datos suficientes para {condition}, trial {trial}, joint {joint_name}")
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        planned_f["time_s"],
        planned_f["desired_position_rad"],
        label="Trayectoria planificada"
    )
    plt.plot(
        actual_f["time_s"],
        actual_f["position_rad"],
        label="Estado articular ejecutado",
        linestyle="--"
    )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Posición articular [rad]")
    plt.title(f"Seguimiento articular: {joint_name} ({condition}, repetición {trial})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"seguimiento_{joint_name}_{condition}_trial_{trial}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")


def create_summary_tables(summary, output_dir):
    successful = summary[summary["success"] == 1].copy()

    if successful.empty:
        print("No hay ejecuciones exitosas para generar resumen.")
        return

    grouped = successful.groupby("condition").agg(
        repeticiones_exitosas=("success", "sum"),
        tiempo_planificacion_prom_s=("planning_time_s", "mean"),
        tiempo_planificacion_std_s=("planning_time_s", "std"),
        tiempo_ejecucion_prom_s=("execution_time_s", "mean"),
        tiempo_ejecucion_std_s=("execution_time_s", "std"),
        duracion_planificada_prom_s=("planned_duration_s", "mean"),
        puntos_trayectoria_prom=("trajectory_points", "mean"),
        error_cartesiano_prom_mm=("cartesian_error_mm", "mean"),
        error_cartesiano_std_mm=("cartesian_error_mm", "std"),
        error_articular_prom_rad=("joint_error_rad", "mean"),
        error_articular_std_rad=("joint_error_rad", "std"),
    ).reindex(CONDITION_ORDER)

    csv_path = os.path.join(output_dir, "summary_grouped_metrics.csv")
    tex_path = os.path.join(output_dir, "summary_grouped_metrics.tex")

    grouped.to_csv(csv_path)
    grouped.round(4).to_latex(tex_path)

    print(f"Guardado: {csv_path}")
    print(f"Guardado: {tex_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/tmp/h1_2_results")
    parser.add_argument("--output-dir", default="/tmp/h1_2_results/figures")
    parser.add_argument("--tracking-condition", default="P2_intermedia")
    parser.add_argument("--tracking-trial", type=int, default=1)
    parser.add_argument("--tracking-joint", default="left_elbow_joint")

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    summary_path = os.path.join(args.input_dir, "experiment_summary.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No se encontró: {summary_path}")

    summary = pd.read_csv(summary_path)

    create_summary_tables(summary, args.output_dir)

    plot_bar(
        summary,
        metric="planning_time_s",
        ylabel="Tiempo de planificación [s]",
        title="Tiempo de planificación por condición experimental",
        output_path=os.path.join(args.output_dir, "tiempo_planificacion.png")
    )

    plot_bar(
        summary,
        metric="execution_time_s",
        ylabel="Tiempo de ejecución [s]",
        title="Tiempo de ejecución por condición experimental",
        output_path=os.path.join(args.output_dir, "tiempo_ejecucion.png")
    )

    plot_bar(
        summary,
        metric="cartesian_error_mm",
        ylabel="Error cartesiano final [mm]",
        title="Error cartesiano final por condición experimental",
        output_path=os.path.join(args.output_dir, "error_cartesiano_final.png")
    )

    plot_bar(
        summary,
        metric="trajectory_points",
        ylabel="Número de puntos",
        title="Puntos de trayectoria por condición experimental",
        output_path=os.path.join(args.output_dir, "puntos_trayectoria.png")
    )

    plot_joint_tracking(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        condition=args.tracking_condition,
        trial=args.tracking_trial,
        joint_name=args.tracking_joint
    )


if __name__ == "__main__":
    main().add_argument("--output-dir", default="/tmp/h1_2_results/figures")
    parser.add_argument("--tracking-condition", default="P2_intermedia")
    parser.add_argument("--tracking-trial", type=int, default=1)
    parser.add_argument("--tracking-joint", default="left_elbow_joint")

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    summary_path = os.path.join(args.input_dir, "experiment_summary.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No se encontró: {summary_path}")

    summary = pd.read_csv(summary_path)

    create_summary_tables(summary, args.output_dir)

    plot_bar(
        summary,
        metric="planning_time_s",
        ylabel="Tiempo de planificación [s]",
        title="Tiempo de planificación por condición experimental",
        output_path=os.path.join(args.output_dir, "tiempo_planificacion.png")
    )

    plot_bar(
        summary,
        metric="execution_time_s",
        ylabel="Tiempo de ejecución [s]",
        title="Tiempo de ejecución por condición experimental",
        output_path=os.path.join(args.output_dir, "tiempo_ejecucion.png")
    )

    plot_bar(
        summary,
        metric="cartesian_error_mm",
        ylabel="Error cartesiano final [mm]",
        title="Error cartesiano final por condición experimental",
        output_path=os.path.join(args.output_dir, "error_cartesiano_final.png")
    )

    plot_bar(
        summary,
        metric="trajectory_points",
        ylabel="Número de puntos",
        title="Puntos de trayectoria por condición experimental",
        output_path=os.path.join(args.output_dir, "puntos_trayectoria.png")
    )

    plot_joint_tracking(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        condition=args.tracking_condition,
        trial=args.tracking_trial,
        joint_name=args.tracking_joint
    )


if __name__ == "__main__":
    main()
