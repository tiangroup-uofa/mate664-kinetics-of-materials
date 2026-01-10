import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(A_slider, Ea_slider, arrhenius_plot, mo):
    _A = A_slider.value  # "global variable"
    _Ea_kJ = Ea_slider.value  # "global variable"
    _ax = arrhenius_plot(_A, _Ea_kJ)
    mo.vstack(
        [
            mo.md(r"""## Example of Thermodynamic Driving Force

                Arrhenius rate law: A --> B \(\displaystyle k = ν\,\exp\!\left(-\frac{E_a}{RT}\right)\)"""),
            mo.hstack([A_slider, Ea_slider], justify="start"),
            _ax,
        ]
    )
    return


@app.cell
def _(mo):
    # A simple demonstration of Arrhenius equation
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Interactive controls (global-style variables) ---
    A_slider = mo.ui.slider(
        start=0.1, stop=10.0, step=0.1, value=1.0,
        label="Pre-exponential factor ν (arb.)"
    )
    Ea_slider = mo.ui.slider(
        start=0.0, stop=200.0, step=1.0, value=60.0,
        label="Activation energy Ea (kJ/mol)"
    )


    def arrhenius_plot(A, Ea_kJ):
        # --- Arrhenius rate (arbitrary units) ---
        R = 8.314          # J/mol/K
        T = 298.15         # K
        Ea = Ea_kJ * 1e3   # J/mol
        k = A * np.exp(-Ea / (R * T))

        # --- Reaction coordinate: A (0) -> TS (1) -> B (2) ---
        x = np.linspace(-0.5, 3.0, 600)

        # Energy levels for A and B (arbitrary but fixed)
        E_A = 0.0
        E_B = -0.35

        # Barrier height above A determined by Ea (visual mapping, arbitrary)
        # Keep this mapping stable and monotonic with Ea_kJ.
        barrier_height = 0.6 + Ea_kJ / 60.0
        E_TS = E_A + barrier_height  # TS energy

        # Piecewise quadratic (parabolic) segments:
        # Segment 1: parabola from A at x=0 to TS at x=1 (vertex at x=1)
        # Segment 2: parabola from TS at x=1 to B at x=2 (vertex at x=1)
        E = np.full_like(x, np.nan, dtype=float)

        # Left segment: x in [0, 1]
        mask1 = (x >= 0.0) & (x <= 1.0)
        # E(x) = E_TS - a*(x-1)^2, with E(0)=E_A -> a = E_TS - E_A
        a1 = (E_TS - E_A)
        E[mask1] = E_TS - a1 * (x[mask1] - 1.0)**2

        # Right segment: x in [1, 2]
        mask2 = (x >= 1.0) & (x <= 2.0)
        # E(x) = E_TS - a*(x-1)^2, with E(2)=E_B -> a = E_TS - E_B
        a2 = (E_TS - E_B)
        E[mask2] = E_TS - a2 * (x[mask2] - 1.0)**2

        # For x outside [0,2], keep flat extensions at A and B for visual continuity
        mask_left_ext = x < 0.0
        mask_right_ext = x > 2.0
        E[mask_left_ext] = E_A
        E[mask_right_ext] = E_B

        # Key points
        x_pts = np.array([0.0, 1.0, 2.0])
        E_pts = np.array([E_A, E_TS, E_B])

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(x, E, linewidth=2)
        ax.scatter(x_pts, E_pts, s=90, zorder=5)

        # Horizontal bars for A and B energy levels
        ax.hlines(E_A, -0.35, 0.35, linewidth=3, alpha=0.9)
        ax.hlines(E_B,  1.65, 2.35, linewidth=3, alpha=0.9)

        # Labels
        ax.annotate("A", (0.0, E_A), xytext=(-10, 12), textcoords="offset points", ha="right")
        ax.annotate("TS", (1.0, E_TS), xytext=(0, 12), textcoords="offset points", ha="center")
        ax.annotate("B", (2.0, E_B), xytext=(10, 12), textcoords="offset points", ha="left")

        ax.set_xlim(-0.5, 3.0)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["A", "TS", "B"])
        ax.set_ylabel("Energy (arb.)")

        # Fixed y-axis scale (choose bounds that comfortably cover slider ranges)
        # Adjust these if your Ea slider range changes.
        ax.set_ylim(-0.9, 5.0)

        ax.set_title(f"Arrhenius rate:  k = A·exp(-Ea/RT)   →   k ≈ {k:.3e} (arb.)")
        ax.grid(True, alpha=0.25)

        return ax
    return A_slider, Ea_slider, arrhenius_plot


if __name__ == "__main__":
    app.run()
