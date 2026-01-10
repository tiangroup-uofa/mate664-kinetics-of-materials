import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    return np, plt


@app.cell
def _(mo, np, plt):
    def solve_K1_K2(z1, z2, xA1, xA2, s):
        """
        Solve for K1 and K2 in x_A = s - K1 exp(K2 z)
        given x_A(z1)=xA1 and x_A(z2)=xA2
        """
        K2 = np.log((s - xA2) / (s - xA1)) / (z2 - z1)
        K1 = (s - xA1) / np.exp(K2 * z1)
        return K1, K2

    def xa_profile(z1, z2, xA1, xA2, s):
        K1, K2 = solve_K1_K2(z1, z2, xA1, xA2, s)
        z_arr = np.linspace(z1, z2, 100)
        x_arr = s - K1 * np.exp(K2 * z_arr)
        return z_arr, x_arr

    def gen_solution_ratio(xA1, xA2, s):
        emcd = xA1 - xA2
        gen = s * np.log((s - xA2)/(s - xA1))
        return gen / emcd

    def plot_xa_profile(xA1=0.75, xA2=0.25, s=0.5, z1=0.0, z2=1.0):
        z_arr, x_arr = xa_profile(z1, z2, xA1, xA2, s)
        plt.figure()
        plt.plot(z_arr, x_arr, label=r"$x_A(z)$ General Solution")
        if np.any(np.isnan(x_arr)):
            plt.text(x=0.5, y=0.5, s="No Real Solution!!", color="red")
        plt.scatter([z1, z2], [xA1, xA2], zorder=3, label="Boundary values")
        plt.plot([z1, z2], [xA1, xA2], "--", label="EMCD case")
        plt.xlabel("z (Abstract Unit)")
        plt.ylabel(r"$x_A$")
        plt.axhline(1.0, ls="--", color="gray")
        plt.text(x=0.02, y=1.005, s="$x_{A} + x_{B} = 1$", va="bottom")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.title("General Solution Profile for Gas Diffusion")
        return plt.gca()

    xa1_slider = mo.ui.slider(0.0, 1.0, 0.05, value=0.75, show_value=True, label="$x_{A1}$")
    xa2_slider = mo.ui.slider(0.0, 1.0, 0.05, value=0.25, show_value=True, label="$x_{A2}$")
    k_slider = mo.ui.slider(-10, 10, 0.1, value=-1.0, show_value=True, label="$k = N_B / N_A$")
    return (
        gen_solution_ratio,
        k_slider,
        plot_xa_profile,
        xa1_slider,
        xa2_slider,
    )


@app.cell
def _(
    gen_solution_ratio,
    k_slider,
    mo,
    plot_xa_profile,
    xa1_slider,
    xa2_slider,
):
    _s = 1/(1+(k_slider.value + 1e-5))
    _xA1 = xa1_slider.value
    _xA2 = xa2_slider.value
    _ratio = gen_solution_ratio(_xA1, _xA2, _s)
    mo.vstack([mo.hstack([xa1_slider, xa2_slider, k_slider], justify="start"),
               mo.md("EMCD: $x_A{z}$ linear; $N_A \\propto x_{A2} - x_{A1}$"),
               mo.md(f"$s$={_s:.3f}, $N_A$ (general) / $N_A$ (EMCD) = {_ratio:.3f}"),
    plot_xa_profile(xA1=xa1_slider.value, xA2=xa2_slider.value, s=_s, 
                    z1=0.0, z2=1.0)
    ])

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
