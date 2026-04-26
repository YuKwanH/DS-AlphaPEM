import matplotlib.pyplot as plt
from config.settings import *
from model.coefficients import *

def display(solution, model):
    profile_1d = {"v": [], "O2": [], "H2": [], "saturation": [], "lambda": [], "T": []}
    for name in nodes_names_vp:
        profile_1d["v"].append(solution.y[model.variable_names.index(name), -1])
    for name in nodes_name_O2:
        profile_1d["O2"].append(solution.y[model.variable_names.index(name), -1])
    for name in nodes_names_H2:
        profile_1d["H2"].append(solution.y[model.variable_names.index(name), -1])
    for name in nodes_names_s:
        profile_1d["saturation"].append(solution.y[model.variable_names.index(name), -1])
    for name in nodes_lambda:
        profile_1d["lambda"].append(solution.y[model.variable_names.index(name), -1])
    for name in nodes_T:
        profile_1d["T"].append(solution.y[model.variable_names.index(name), -1])

    profile_panels = [
        ("v", "Vapor Pressure $(mol/m^3)$"),
        ("O2", "Oxygen Concentration $(mol/m^3)$"),
        ("H2", "Hydrogen Concentration $(mol/m^3)$"),
        ("saturation", "Saturation"),
        ("lambda", r"Water Content $(\lambda)$"),
        ("T", "Temperature (K)"),
    ]

    plot_nodes = nodes[1:-1]
    n_panels = len(profile_panels)
    n_cols = 2
    n_rows = (n_panels + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3), sharex=False)
    axes_flat = axes.flatten()

    for ax, (profile_key, title) in zip(axes_flat, profile_panels):
        if profile_key == "v":
            ax.hlines(C_v_sat(operating_inputs['Tfc']), xmin=plot_nodes[0], xmax=plot_nodes[-1], colors='tab:gray', linestyles='--', label="Saturation Vapor Pressure")
        if profile_key == "saturation":
            ax.set_ylim(-0.05, 1.05)
        if profile_key == "lambda":
            ax.set_xlim(borders[1], borders[5])

        y_values = expand_profile_on_nodes(profile_key, profile_1d[profile_key])[1:-1]
        ax.plot(plot_nodes, y_values, linewidth=1.8, marker="o", markersize=3)
        for x in borders:
            ax.axvline(x=x, color="tab:gray", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("value")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    axes_flat[0].legend()
    plt.tight_layout()
    plt.show()