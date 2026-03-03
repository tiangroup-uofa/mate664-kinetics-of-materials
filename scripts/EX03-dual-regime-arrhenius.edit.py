# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.20.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arrhenius Plot Analysis

    The Arrhenius plot shows $\ln D_K$ versus $1/T$ for KCl with varying [CaCl$_2$] concentrations.

    Please consider the following questions:

    - Do you observe two regimes of diffusivitiy in the plot? At which dopant concentration is such effect prominent?
    - Take the slopes of the curve within the tntrinsic and extrinsic regions. Are they matching your expected activation enthalpy values?
    - In many other Arrhenius plots, a steeper slope usually means higher activation energy and thus lower diffusivity. Why does the intrinsic regime show higher $D_K$ instead?
    """)
    return


@app.cell(hide_code=True)
def _(k_B, mo):
    # Default Thermodynamic properties (from Fuller et al.)
    _H_S_f = 2.49  # Schottky defect formation enthalpy (eV)
    _S_S_f = 7.64  # Schottky defect formation entropy (in units of k_B)
    _H_K_m = 0.76  # Cation migration enthalpy (eV)
    _S_K_m = 2.56  # Cation migration entropy (in units of k_B)
    _nu = 6.95e12  # Jumping frequency (s^-1)
    _a = 629.2e-12  # Lattice constant (meters)



    # Precompute entropy terms in eV
    _S_S_f_eV = _S_S_f * k_B
    _S_K_m_eV = _S_K_m * k_B

    setups = mo.md("""
    ## Define Physical Quantities

    The following quantities should be set to make the Arrhenius plot. The default values are taken from Fuller et al (1968). 
    But you're free to change accordingly.

    - Schottky vacancy formation: $H_{{S}}^{{f}}$ (eV) {H_S_f}, $S_{{S}}^{{f}}$ ($k_B$) {S_S_f}
    - Cation migration: $H_{{K}}^{{m}}$ (eV) {H_K_m}, $S_{{K}}^{{m}}$ ($k_B$) {S_K_m}
    - Lattice parameter: $a$ (pm) {latt_a}, Jumpting frequency: $\\nu$ (10$^{{12}}$ s$^{{-1}}$) {nu}

    ## Range in Arrhenius plot

    - Range of dopant log$_{{10}}$ [CaCl$_2$] {conc_range}
    """).batch(
        H_S_f=mo.ui.number(step=0.01, value=_H_S_f),
        S_S_f=mo.ui.number(step=0.01, value=_S_S_f),
        H_K_m=mo.ui.number(step=0.01, value=_H_K_m),
        S_K_m=mo.ui.number(step=0.01, value=_S_K_m),
        latt_a=mo.ui.number(step=0.1, value=_a / 1e-12),
        nu=mo.ui.number(step=0.01, value=_nu / 1e12),
        conc_range=mo.ui.range_slider(start=-12, stop=-2, step=1, value=(-9, -4), show_value=True)
    )
    setups
    return (setups,)


@app.cell(hide_code=True)
def _(f, k_B, np, setups):
    def D_K(T: float | np.ndarray, ca_conc: float, 
            setups: dict = setups.value) -> float | np.ndarray:
        """
        Calculate potassium diffusion coefficient in KCl with CaCl2 dopant.

        This function computes the diffusion coefficient of K+ ions in KCl 
        doped with CaCl2 using defect chemistry principles. The calculation 
        accounts for both intrinsic (thermal) and extrinsic (dopant-induced) 
        potassium vacancies.

        Parameters
        ----------
        T : float or np.ndarray
            Temperature in Kelvin
        ca_conc : float  
            CaCl2 concentration in mole fraction
        setups : dict, optional
            Dictionary containing thermodynamic parameters:
            - H_S_f: Schottky defect formation enthalpy (eV)
            - S_S_f: Schottky defect formation entropy (k_B units)
            - H_K_m: Cation migration enthalpy (eV)
            - S_K_m: Cation migration entropy (k_B units)  
            - latt_a: Lattice parameter (pm)
            - nu: Jump frequency (10^12 s^-1)

        Returns
        -------
        float or np.ndarray
            Potassium diffusion coefficient in m^2/s

        Notes
        -----
        The calculation follows these steps:
        1. Compute intrinsic K+ vacancy concentration from Schottky equilibrium
        2. Calculate total K+ vacancy concentration including Ca2+ dopant effects
        3. Apply Einstein relation: D = [V_K'] * f * λ² * ν * exp(-ΔG_m/kT)

        The dopant creates additional K+ vacancies via charge compensation:
        Ca2+ on K+ site requires 1 K+ vacancy for charge neutrality.
        """
        # Calculate Gibbs free energy for Schottky defect formation
        G_S_f = setups["H_S_f"] - setups["S_S_f"] * k_B * T

        # Intrinsic K+ vacancy concentration from thermal equilibrium
        # [V_K']_pure ^2 = exp(-ΔG_S_f / kT)
        V_K_pure = np.exp(-G_S_f / 2 / k_B / T)

        # Total K+ vacancy concentration including dopant effects and equilibrium
        V_K_total = ca_conc / 2 * (1 + np.sqrt(1 + (2 * V_K_pure / ca_conc) ** 2))

        # Calculate Gibbs free energy for K+ migration
        G_K_m = setups["H_K_m"] - T * setups["S_K_m"] * k_B

        # Jump distance for K+ diffusion (nearest neighbor distance)
        # For rocksalt structure: λ = a / √2
        lambda_jump = setups["latt_a"] * 1e-12 / np.sqrt(2)  # Convert pm to m

        # Attempt frequency for K+ jumps
        nu = setups["nu"] * 1e12  # Convert from 10^12 s^-1 to s^-1

        # Einstein relation for diffusion coefficient
        # D_K = [V_K'] * f * λ² * ν * exp(-ΔG_m / kT)
        # where f is correlation factor for vacancy mechanism
        D_K_val = (V_K_total * f * lambda_jump**2 * nu * 
                   np.exp(-G_K_m / (k_B * T)))

        return D_K_val

    return (D_K,)


@app.cell(hide_code=True)
def _(D_K, np, plt, setups):
    def plot_arrhenius(setups_value: dict = setups.value) -> plt.Axes:
        """Prepare Arrhenius plot: ln(D_K) vs 1/T for different CaCl2 concentrations"""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Range of CaCl2 concentrations (mol fraction)
        start, stop = setups_value["conc_range"]
        _ca_conc_range = 10.0 ** np.arange(start, stop + 0.1, 1)[::-1]

        # Temperature range (K)
        _T_range = np.linspace(400, 1000, 100)

        for ca_conc in _ca_conc_range:
            D_vals = D_K(_T_range, ca_conc, setups_value)
            ax.plot(1 / _T_range, np.log(D_vals), label=f"[CaCl2]={ca_conc:.0e}")

        # Extra line for pure intrinsic range
        D_pure = D_K(_T_range, 1e-30, setups_value)
        ax.plot(
            1 / _T_range,
            np.log(D_pure),
            ls="--",
            color="grey",
            alpha=0.8,
            label="Intrinsic limit",
        )
        ax.set_ylim(-55, -30)

        ax.set_xlabel(r"$1/T$ (1/K)")
        ax.set_ylabel(r"$\ln [D_K / (m^2/s)]$")
        ax.set_title(
            r"Arrhenius Plot of $\ln D_K$ vs $1/T$ for KCl with varying "
            r"[CaCl$_2$]"
        )
        ax.legend(title="[CaCl$_2$]")
        ax.grid(True)

        # Add right y-axis with log10 values
        ax_right = ax.twinx()
        ax_right.set_ylabel(r"$D_K$ (m$^2$/s)")
        # Convert ln to log10: log10(x) = ln(x) / ln(10)
        y_min, y_max = ax.get_ylim()
        ax_right.set_ylim(y_min / np.log(10), y_max / np.log(10))

        # Format right y-axis tick labels as 10^{value}
        right_ticks = ax_right.get_yticks()
        ax_right.set_yticklabels([f"$10^{{{tick:.0f}}}$" for tick in right_ticks])

        # Add top axis with temperature values
        ax2 = ax.twiny()
        temp_ticks = np.arange(_T_range.max(), _T_range.min() - 1, -100)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(1 / temp_ticks)
        ax2.set_xticklabels([f"{t:.0f}" for t in temp_ticks])
        ax2.set_xlabel(r"$T$ (K)")

        return ax


    ax = plot_arrhenius(setups.value)
    ax
    return


@app.cell(hide_code=True)
def _():
    # Physical constants
    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
    f = 0.7  # Correlated jump correction factor
    return f, k_B


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


if __name__ == "__main__":
    app.run()
