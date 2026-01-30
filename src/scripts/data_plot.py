import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def main(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # --- Load CSV ---
    df = pd.read_csv(csv_path)

    # Normalize time to start at zero
    t0 = df["t"].iloc[0]
    df["t_rel"] = df["t"] - t0

    # Convert meters ? millimeters for readability
    df["error_x_mm"] = df["error_x_m"] * 1000
    df["error_z_mm"] = df["error_z_m"] * 1000
    df["err_mag_mm"] = df["err_mag"] * 1000
    df["control_x_mm"] = df["control_x"] * 1000
    df["control_z_mm"] = df["control_z"] * 1000

    # --- Plot 1: XZ error convergence ---
    plt.figure()
    plt.plot(df["t_rel"], df["error_x_mm"], label="X error [mm]")
    plt.plot(df["t_rel"], df["error_z_mm"], label="Z error [mm]")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [mm]")
    plt.title("Visual Servoing Error (XZ)")
    plt.legend()
    plt.grid(True)

    # --- Plot 2: Error magnitude ---
    plt.figure()
    plt.plot(df["t_rel"], df["err_mag_mm"], label="||error|| XZ [mm]")
    plt.axhline(5, linestyle="--", color="r", label="Stability threshold")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude [mm]")
    plt.title("Error Magnitude")
    plt.legend()
    plt.grid(True)

    # --- Plot 3: Control commands ---
    plt.figure()
    plt.plot(df["t_rel"], df["control_x_mm"], label="Control X [mm]")
    plt.plot(df["t_rel"], df["control_z_mm"], label="Control Z [mm]")
    plt.xlabel("Time [s]")
    plt.ylabel("Command [mm]")
    plt.title("Control Commands")
    plt.legend()
    plt.grid(True)

    # --- Plot 4: PLC state timeline ---
    plt.figure()
    plt.step(df["t_rel"], df["frame"], where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("PLC State")
    plt.title("PLC State Over Time")
    plt.grid(True)

    # --- Highlight first stable moment ---
    stable_rows = df[df["stable"] == True]
    if not stable_rows.empty:
        t_stable = stable_rows["t_rel"].iloc[0]
        for fig_num in plt.get_fignums():
            plt.figure(fig_num)
            plt.axvline(t_stable, linestyle="--", color="g", alpha=0.7)

    output_dir = "plots"
    Path(output_dir).mkdir(exist_ok=True)

    for i in plt.get_fignums():
	    plt.figure(i)
	    plt.savefig(f"{output_dir}/figure_{i}.png", dpi=300, bbox_inches="tight")

    print("Plots saved to ./plots/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_trial.py trial_001.csv")
        sys.exit(1)

    main(sys.argv[1])
