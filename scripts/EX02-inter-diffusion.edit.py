import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 2 Q3 Interactive Demo
    ## Finite Difference (FD) Method For Interdiffusion Simulation

    In this demo we will show how to solve an interdiffusion problem between two diffusion couple (1-2),
    with varying interdiffusivity given by the Darken equation

    $$
    \tilde{D} = D_1 X_2 + D_2 X_1
    $$

    We have already provided an implementation of the FD scheme and boundary
    conditions in the `calculate_interdiffusion` function,
    which defines the numerical ODEs used to solve the $X_1(x, t)$ profile use `scipy.integrate.solve_ivp`.

    ## Compare the $X_1$ profile using varying and fixed $\tilde{D}$ values

    We should be able to see the different between the diffusion profile under these two conditions:
    1. Use the exact position-dependent interdiffusivity, i.e. $\tilde{D} = \tilde{D}(x)$
    2. Use the averaged $\tilde{D}=D_1 (1- \overline{X}_1) + D_2 \overline{X}_1$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    log10_D1_slider = mo.ui.slider(-16, -8, 0.1, value=-10.5, label=r"$\text{log}_{10} D_1/(\text{m}^2/\text{s})$", show_value=True)
    log10_D2_slider = mo.ui.slider(-16, -8, 0.1, value=-9.5, label=r"$\text{log}_{10} D_2/(\text{m}^2/\text{s})$", show_value=True)
    X1L_slider = mo.ui.slider(0, 1, 0.05, value=0.6, label=r"$X_{1L}$ (Zn)", show_value=True)
    X1R_slider = mo.ui.slider(0, 1, 0.05, value=0.0, label=r"$X_{1R}$ (Zn)", show_value=True)
    time_slider = mo.ui.slider(1, 30, 0.5, value=7, label="Time (days)", show_value=True)
    grid_slider = mo.ui.slider(5, 200, value=50, label="x-grid points", step=1, show_value=True)
    return (
        X1L_slider,
        X1R_slider,
        log10_D1_slider,
        log10_D2_slider,
        time_slider,
    )


@app.cell(hide_code=True)
def _(
    X1L_slider,
    X1R_slider,
    log10_D1_slider,
    log10_D2_slider,
    mo,
    time_slider,
):
    question1 = mo.md("""
    Assume we have a Zn-Cu diffusion couple where Zn=1 and Cu=2. 
    Initially we have a Zn-Cu alloy with $X_{{1L}}$ Zinc at left and $X_{{1R}}$ Zinc at right. How will the $X_1(x)$ profile look using the two assumptions?

    Please choose the values for:
    - {logd1} {logd2}
    - {X1L}  {X1R}
    - {time}

    Observe the plot, you should see the exact $\\tilde{{D}}$ method reproduce an asymmetric $X_1(x)$ profile. How will the profile change with the relation with $D_1$ and $D_2$?


    """).batch(logd1=log10_D1_slider,
              logd2=log10_D2_slider,
              X1L=X1L_slider, X1R=X1R_slider,
              time=time_slider,
              )
    return (question1,)


@app.cell(hide_code=True)
def _(mo, question1, show_plot):
    mo.vstack([question1, show_plot()], justify="start")
    return


@app.cell(hide_code=True)
def ode_def(np, solve_ivp):
    def calculate_interdiffusion(
        log10_D1: float,
        log10_D2: float,
        t_days: float = 7.0,
        grid_points: int = 200,
        X1_L: float = 0.8,
        X1_R: float = 0.2,
        use_average = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate interdiffusion of two metals using finite difference method.

        Args:
            log10_D1: Log10 of diffusivity of metal 1 (m²/s)
            log10_D2: Log10 of diffusivity of metal 2 (m²/s)
            t_days: Simulation time in days
            grid_points: Number of points in spatial grid
            X1_L: Initial molar fraction of metal 1 on left side
            X1_R: Initial molar fraction of metal 1 on right side
            L: Length of domain in meters
            use_average: Use the "averaged" interdiffusivity

        Returns:
            Tuple of (x grid, time points, concentration matrix, interdiffusivity history)
        """
        # Convert parameters
        D1 = 10**log10_D1  # m²/s
        D2 = 10**log10_D2  # m²/s

        # "Wrong way to calculate average X1 and D
        X1_avg = (X1_L + X1_R) / 2
        D_avg = D1 * (1 - X1_avg) + D2 * X1_avg

        t_final = t_days * 24 * 3600  # Convert days to seconds
        # We always take the L-domain size according to the diffusion length scale
        L = np.sqrt(4 * max(D1, D2) * t_final) * 2

        # Set up spatial grid
        x = np.linspace(-L, L, grid_points)
        dx = x[1] - x[0]

        # Initial condition
        X1_0 = np.ones(grid_points)
        X1_0[: grid_points // 2] = X1_L
        X1_0[grid_points // 2 :] = X1_R

        # Define the RHS function for the ODE system
        def diffusion_rhs(t: float, X1: np.ndarray) -> np.ndarray:
            # Ensure X1 is bounded between 0 and 1
            X1 = np.clip(X1, 0, 1)
            X2 = 1 - X1

            # Calculate interdiffusivity at each point
            # We need the interdiffusivity at the central to calculate the flux
            if use_average:
                D = np.ones_like(X1) * D_avg
            else:
                D = D1 * X2 + D2 * X1

            # Initialize the derivative array
            dX1dt = np.zeros_like(X1)

            # Finite difference for interior points
            # integrate without the boundary
            for i in range(1, len(X1) - 1):
                # Calculate D*dX1/dx at i+1/2
                D_right = 0.5 * (D[i] + D[i + 1])
                dX1dx_right = (X1[i + 1] - X1[i]) / dx

                # Calculate D*dX1/dx at i-1/2
                D_left = 0.5 * (D[i - 1] + D[i])
                dX1dx_left = (X1[i] - X1[i - 1]) / dx

                # Second derivative using central difference
                d2X1dx2 = (D_right * dX1dx_right - D_left * dX1dx_left) / dx
                dX1dt[i] = d2X1dx2

            # The boundaries are Dirichlet (const X1 at begin and end)
            dX1dt[0] = 0
            dX1dt[-1] = 0

            return dX1dt

        # Solve the system using solve_ivp
        t_eval = np.linspace(0, t_final, 10)
        solution = solve_ivp(
            diffusion_rhs,
            [0, t_final],
            X1_0,
            method="BDF",
            t_eval=t_eval,
            rtol=1e-5,
            atol=1e-8,
        )
    
        # Calculate interdiffusivity for each timestep and position
        # First clip X1 values to ensure they're in [0, 1]
        y_clipped = np.clip(solution.y, 0, 1)
        # Calculate X2 = 1 - X1
        X2 = 1 - y_clipped
        # Calculate D for each X1 value
        if use_average:
            D_history = np.ones_like(y_clipped) * D_avg
        else:
            D_history = D1 * X2 + D2 * y_clipped
    
        return x, solution.t, solution.y.T, D_history.T
    return (calculate_interdiffusion,)


@app.cell(hide_code=True)
def _(calculate_interdiffusion, plt, question1):
    def show_plot():
        log10d1, log10d2 = question1.value["logd1"], question1.value["logd2"]
        t_days = question1.value["time"]
        X1L = question1.value["X1L"]
        X1R = question1.value["X1R"]
        x, sol_t, sol_y, sol_D = calculate_interdiffusion(
            log10_D1=log10d1,
            log10_D2=log10d2,
            t_days=t_days,
            X1_L=X1L,
            X1_R=X1R,
            use_average=False,
        )
        x_avg, sol_t_avg, sol_y_avg, _ = calculate_interdiffusion(
            log10_D1=log10d1,
            log10_D2=log10d2,
            t_days=t_days,
            X1_L=X1L,
            X1_R=X1R,
            use_average=True,
        )
        plt.subplots(1, 2, figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(x, sol_y[-1], label=r"Exact $\tilde{D}$")
        plt.plot(x, sol_y_avg[-1], "--", label=r"Averaged $\tilde{D}$")
        plt.axvline(x=0.0, ls="-", color="grey", alpha=0.5)
        plt.text(
            x=0.0, y=plt.ylim()[1], s="Initial boundary", ha="center", va="bottom"
        )
        plt.axhline(y=(X1L + X1R) / 2, ls="-", color="grey", alpha=0.5)
        plt.text(
            y=(X1L + X1R) / 2,
            x=plt.xlim()[1],
            s=r"$\frac{X_{1L} + X_{1R}}{2}$",
            ha="left",
        )
        plt.xlabel("Position $x$ (m)")
        plt.ylabel("$X_1(x)$ (Zn frac)")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, sol_D[-1], label="$\\tilde{{D}}(x)$")
        plt.xlabel("Position $x$ (m)")
        plt.ylabel("Inter-Diffusivity (m$^2$/s)")
        plt.yscale("log")
        plt.grid("on")
        plt.legend()

        plt.tight_layout()
        return plt.gca()
    return (show_plot,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.special import erf, erfc
    return np, plt, solve_ivp


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
