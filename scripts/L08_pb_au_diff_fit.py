# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import erf
    from scipy.optimize import minimize
    return erf, minimize, mo, np, plt


@app.cell
def _(np):
    # Molar masses (g/mol)
    M_Au = 196.97
    M_Pb = 207.2
    # Experimental section length
    h_cm = 1.054
    Delta_cm = 2.018

    dist_cm_exp = (np.arange(0, 13) + 0.5) * h_cm
    dist_m_exp = dist_cm_exp / 100

    # Weight percentage * 100%
    wt_Au_exp = np.array([7.38, 7.26, 6.87, 6.29, 5.60, 4.92, 4.25, 3.58, 2.87, 2.24, 1.75, 1.36, 0.97])  # in wt%
    x_Au_exp = wt_Au_exp / M_Au / (wt_Au_exp / M_Au + (1 - wt_Au_exp) / M_Pb)

    # Experimental conditions
    t_exp = 3600 * 24 * 6.96  # 7 days in seconds
    T_exp = 492 + 273.15  # 600 C in Kelvin
    return Delta_cm, M_Au, M_Pb, dist_m_exp, t_exp, wt_Au_exp


@app.cell
def _(mo):
    # UI components
    D_input = mo.ui.number(start=-12, stop=-8, value=-9, step=0.005, label="log10 D (m²/s)")
    x0_input = mo.ui.number(start=0.01, stop=0.99, value=0.35, step=0.005, label="Au molar fraction (t=0)")
    show_lsq = mo.ui.switch(value=False, label="Show least square fitting")
    return D_input, show_lsq, x0_input


@app.cell
def _(D_input, mo, plot_fitting_results, show_lsq, x0_input):
    mo.vstack([mo.md("""Roberts-Austen Experiment
    """),
    mo.hstack([D_input, x0_input, show_lsq], wrap=True, justify="start"),
    plot_fitting_results()]
    )
    return


@app.cell
def _(
    D_input,
    Delta_cm,
    M_Au,
    M_Pb,
    dist_m_exp,
    erf,
    minimize,
    np,
    plt,
    show_lsq,
    t_exp,
    wt_Au_exp,
    x0_input,
):
    def thin_slab_solution(x: np.ndarray, D: float,
                           x0: float,
                           t_max: float = t_exp, 
                           thickness: float = Delta_cm / 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the thin slab solution for diffusion.
    
        Args:
            x: Position in meters
            D: Diffusion coefficient in m²/s
            x0: Initial concentration (mole fraction)
            t_max: Maximum time in seconds
            thickness: Thickness of the slab in meters
        
        Returns:
            Tuple of (mole_fraction, weight_percent)
        """
        res = x0 / 2 * (erf((x + thickness) / (np.sqrt(4 * D * t_max))) - 
                       erf((x - thickness) / (np.sqrt(4 * D * t_max))))
        res_wt = 100 * (res * M_Au) / (res * M_Au + (1 - res) * M_Pb)
        return res, res_wt


    def fit_diffusion_profile(
        dist_m: np.ndarray, 
        wt_Au: np.ndarray,
        initial_log10D: float = -9.0,
        initial_x0: float = 0.8
    ) -> tuple[float, float]:
        """
        Fit diffusion profile using least squares optimization.
    
        Args:
            dist_m: Distance in meters
            wt_Au: Weight percentage of Au
            initial_log10D: Initial guess for log10(D)
            initial_x0: Initial guess for x0
        
        Returns:
            Tuple of (optimized_D, optimized_x0)
        """
    
    
        def objective(params):
            log10D, x0 = params
            D = 10**log10D
            _, wt_predicted = thin_slab_solution(dist_m, D, x0)
            return np.sum((wt_predicted - wt_Au)**2)
    
        initial_params = [initial_log10D, initial_x0]
        result = minimize(objective, initial_params, method='Nelder-Mead')
        # print(result)
        opt_log10D, opt_x0 = result.x
        opt_D = 10**opt_log10D
    
        return opt_D, opt_x0


    def plot_fitting_results(
        dist_m=dist_m_exp, 
        wt_Au=wt_Au_exp,
        # D: float,
        # x0: float
    ) -> plt.Axes:
        """Plot experimental data and fitted curve"""
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Plot experimental data in original value
        ax.plot(dist_m * 100, wt_Au, 'o', label='Experimental Data')
        ax.set_yscale("log")
    
        # Generate fitted curve
        x_fine = np.linspace(min(dist_m), max(dist_m), 200)
        D = 10 ** D_input.value
        x0 = x0_input.value
        _, wt_fitted = thin_slab_solution(x_fine, D, x0)
    
        # Calculate MSE for manual fitting
        _, wt_manual_pred = thin_slab_solution(dist_m, D, x0)
        manual_mse = np.mean((wt_manual_pred - wt_Au)**2)
    
        # Plot fitted curve with MSE in legend
        ax.plot(x_fine * 100, wt_fitted, '--', 
                label=f'Manual fitting (D={D:.2e} m²/s, MSE={manual_mse:.3f})')

        if show_lsq.value:
            best_D, best_x0 = fit_diffusion_profile(dist_m, wt_Au)
            print(best_D, best_x0)
            _, wt_fitted_lsq = thin_slab_solution(x_fine, best_D, best_x0)
        
            # Calculate MSE for least squares fitting
            _, wt_lsq_pred = thin_slab_solution(dist_m, best_D, best_x0)
            lsq_mse = np.mean((wt_lsq_pred - wt_Au)**2)
        
            ax.plot(x_fine * 100, wt_fitted_lsq, '-', 
                    label=f'Least-Sq fitting (D={best_D:.2e} m²/s, MSE={lsq_mse:.3f})')
    
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Au Weight %')
        ax.set_title('Diffusion Profile Fitting')
        ax.legend()
        ax.set_ylim(0.5, 15)
        ax.grid(True)
    
        return ax
    return (plot_fitting_results,)


@app.cell
def _():
    # plot_fitting_results(dist_m_exp, wt_Au_exp, 1e-9, 0.8)

    return


@app.cell
def _():
    # def fit_and_plot

    # # # Initial and boundary conditions
    # # C0 = x_Au[0]  # left side (pure Au)
    # # C1 = x_Au[-1] # right side (almost pure Pb)

    # # # UI for D (diffusivity)
    # # 
    # # D_input
    # plt.semilogy(dist_cm, wt_Au, "o")
    # res = thin_slab_solution(x=np.linspace(0, 0.15, 100), t_max=t, D=3.4e-9, C0=0.80, Delta=2.108*1e-2/2)
    # plt.semilogy(np.linspace(0, 0.15, 100) * 100, res, "--")
    return


@app.cell
def _():



    return


if __name__ == "__main__":
    app.run()
