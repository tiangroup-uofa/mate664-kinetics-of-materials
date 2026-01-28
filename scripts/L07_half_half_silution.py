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

    # Physical constants
    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K

    # Default parameters
    D0_default = 6e-8  # cm^2/s
    Ea_default = 0.98  # eV
    T_default = 400 + 273.15   # K (typical annealing temp)
    c0_default = 0.6   # Initial Cu concentration on right side

    # UI elements
    log10_D0_slider = mo.ui.slider(-10, -6, value=np.log10(D0_default), label="log10($D_0$) [cm²/s]", show_value=True)
    Ea_slider = mo.ui.slider(0.5, 1.5, value=Ea_default, step=0.01, label="$E_a$ [eV]", show_value=True)
    T_number = mo.ui.number(value=T_default, label="Temperature [K]")
    max_days_slider = mo.ui.slider(1, 120, value=30, label="Max days", show_value=True)


    return (
        Ea_slider,
        T_number,
        c0_default,
        erf,
        k_B,
        log10_D0_slider,
        max_days_slider,
        mo,
        np,
        plt,
    )


@app.cell
def _(Ea_slider, T_number, log10_D0_slider, max_days_slider, mo, plot_D):
    mo.vstack(
        [
            mo.hstack([log10_D0_slider, Ea_slider], justify="start"),
            mo.hstack([T_number, max_days_slider], justify="start"),
            plot_D(),
        ]
    )
    return


@app.cell
def _(
    Ea_slider,
    T_number,
    c0_default,
    erf,
    k_B,
    log10_D0_slider,
    max_days_slider,
    np,
    plt,
):
    def plot_D():
        # Get UI values
        log10_D0 = log10_D0_slider.value
        D0 = 10 ** log10_D0
        Ea = Ea_slider.value
        T = T_number.value
        max_days = max_days_slider.value
    
        # Arrhenius equation for D
        D = D0 * np.exp(-Ea / (k_B * T))
    
        # Display Arrhenius plot
        T_arr = np.linspace(200, 1500, 200) + 273.15
        D_arr = D0 * np.exp(-Ea / (k_B * T_arr))
    
        # plt.figure()
        plt.subplots(1, 2, figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.semilogy(1/T_arr, D_arr, label="Arrhenius fit")
        plt.scatter([1/T], [D], color='red', label=f"T = {T:.0f} K")
        plt.xlabel(r"$1/T$ [1/K]")
        plt.ylabel(r"$D$ [cm$^2$/s]")
        plt.title("Arrhenius Plot for Diffusivity")
        # plt.legend()
        # plt.tight_layout()

        # Concentration profile evolution
        c0 = c0_default
        x = np.linspace(-0.0003, 0.0003, 400)  # cm, symmetric about interface
    
        days = np.linspace(0, max_days, 5)
        times = days * 24 * 3600  # convert days to seconds
    
        # plt.figure(figsize=(8, 5))
        plt.subplot(1, 2, 2)
        for t, day in zip(times, days):
            if t == 0:
                c_xt = np.where(x >= 0, c0, 0)
            else:
                c_xt = c0 / 2 * (1 + erf(x / np.sqrt(4 * D * t)))
            plt.plot(x * 1e4, c_xt, label=f"{day:.1f} days")
        plt.xlabel("x [μm]")
        plt.ylabel("Cu Concentration (fraction)")
        plt.title("Diffusion Couple: Pure Ni | Cu$_{0.6}$Ni$_{0.4}$")
        plt.legend()
        plt.tight_layout()
        plt.gca()
        ax = plt.gca()
        return ax
    return (plot_D,)


@app.cell
def _():


    return


@app.cell
def _(mo):
    mo.md(r"""
    # Diffusion Couple: Pure Ni and Cu$_{0.6}$Ni$_{0.4}$

    This notebook simulates the interdiffusion between a semi-infinite pure Nickel (left, $x<0$) and a Cu$_{0.6}$Ni$_{0.4}$ alloy (right, $x>0$) using the error function solution:

    $$c(x, t) = \frac{c_0}{2}\left[1 + \operatorname{erf}\left(\frac{x}{\sqrt{4Dt}}\right)\right]$$

    - **$D$** is the interdiffusion coefficient, calculated by the Arrhenius equation:

    $$D = D_0 \exp\left(-\frac{E_a}{k_B T}\right)$$

    - **$D_0$**: Pre-exponential factor (cm$^2$/s), **$E_a$**: Activation energy (eV), **$T$**: Temperature (K)
    - Adjust $\log_{10}(D_0)$, $E_a$, and max days to see the evolution of the concentration profile.
    - The Arrhenius plot shows $D$ as a function of $1/T$.
    """)
    return


if __name__ == "__main__":
    app.run()
