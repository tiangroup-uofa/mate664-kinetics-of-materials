# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.special import erf, erfc

    return erfc, mo, np, plt, solve_ivp


@app.cell
def _(mo):
    # UI elements for user interaction
    log10_D_slider = mo.ui.slider(-11, -9, 0.1, value=-9.5, label="log10 D", show_value=True)
    time_slider = mo.ui.slider(1, 30, 0.5, value=7, label="Time (days)", show_value=True)
    grid_slider = mo.ui.slider(5, 200, value=50, label="x-grid points", step=1, show_value=True)

    # mo.hstack([log10_D_slider, time_slider, grid_slider])
    return grid_slider, log10_D_slider, time_slider


@app.cell
def _(erfc, np, plt, solve_ivp):
    def diffusion_fd_vs_analytical(log10_D: float, t_days: float, N: int) -> plt.Axes:
        # Parameters
        t = 3600 * 24 * t_days
        D = 10 ** log10_D
        #L = 1.0  # domain length (arbitrary, since semi-infinite)
        Ld = np.sqrt(4 * D * t)
        L = 5 * Ld
        x = np.linspace(0, L, N)
        dx = x[1] - x[0]
        C0 = 1.0  # wall concentration
        C_init = np.zeros(N)

        # MOL ODE system: dC/dt = D * d2C/dx2
        def rhs(t: float, C: np.ndarray) -> np.ndarray:
            dCdt = np.zeros_like(C)
            # Boundary at x=0: constant wall concentration
            dCdt[0] = 0.0
            # Interior points
            dCdt[1:-1] = D * (C[2:] - 2*C[1:-1] + C[:-2]) / dx**2
            # Neumann BC at x=L (semi-infinite): dC/dx = 0
            dCdt[-1] = D * (C[-2] - C[-1]) / dx**2
            return dCdt

        # Initial condition: all zero
        C0_vec = np.zeros(N)
        C0_vec[0] = C0

        # Enforce wall BC at x=0 for all time
        def event_wall(t: float, C: np.ndarray) -> float:
            C[0] = C0
            return 0

        # Integrate
        sol = solve_ivp(rhs, [0, t], C0_vec, method='RK45', t_eval=[t], vectorized=False)
        C_fd = sol.y[:, -1].copy()
        C_fd[0] = C0  # enforce wall BC

        # Analytical solution (semi-infinite, constant wall):
        # C(x, t) = C0 * erf(x / (2*sqrt(D*t)))
        x_ana = np.linspace(0, L, 200)

        C_analytical = C0 * erfc(x_ana / (2 * np.sqrt(D * t)))

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x / 1e-6, C_fd, 'o-', label='FD (MOL)', markersize=7)
        ax.plot(x_ana / 1e-6, C_analytical, '-', label='Analytical', linewidth=2)
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('c/c₀')
        ax.set_title(f'D={D:.2e} m²/s, t={t_days} days')
        ax.legend()
        ax.grid(True)
        return ax
    return (diffusion_fd_vs_analytical,)


@app.cell
def _(
    diffusion_fd_vs_analytical,
    grid_slider,
    log10_D_slider,
    mo,
    time_slider,
):
    # Display UI and plot together
    mo.vstack([
        mo.md("""- Problem: diffusion into semi-infinite space from a constant wall concentration.
        - Analytical solution: superimposition + reflection

        $$
        c(x, t) = c_0 \\mathrm{erfc}(\\frac{x}{\\sqrt{2 D t}})
        $$
        """),
        mo.hstack([log10_D_slider, time_slider, grid_slider], justify="start", wrap=True),
        diffusion_fd_vs_analytical(
            log10_D_slider.value, time_slider.value, grid_slider.value
        )
    ])
    return


if __name__ == "__main__":
    app.run()
