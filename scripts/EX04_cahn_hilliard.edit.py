import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ## Cahn-Hilliard simulation with regular-solution free energy

    We will simulate the Cahn-Hilliard equation using a more physical potential (adapted from https://web.tuat.ac.jp/~yamanaka/pcoms2019/Cahn-Hilliard-2d.html). The governing CH equation is

    $$
    \frac{\partial x_B}{\partial t} = \nabla \cdot \left(M \nabla \mu_B \right)
    $$

    and

    $$
    \mu_B = \frac{\partial g^{\text{homo}}}{\partial x_B} - 2\kappa \nabla^2 x_B
    $$

    - Gibbs free energy
    $$
    g^{\text{homo}}(x_B) = \omega x_B(1-x_B) + k_B T \left[(1-x_B)\ln(1-x_B)+x_B\ln x_B\right]
    $$

    - Composition-dependent mobility

    $$
    M = \frac{[D_A x_B + D_B (1 - x_B)]}{\frac{k_B T}{x_B(1-x_B)} - 2\omega}
    $$
    """)
    return


@app.cell
def _(mo):
    setup = mo.md("""
    ## Setup the potential profile and initial concentration

    - $\\omega / k_B T$: {omega}
    - $T$ (K): {T}
    - $x_{{B0}}$: {xB0}
    """).batch(
        omega = mo.ui.slider(
            start=0.5, stop=5.0, step=0.05, value=2.25, show_value=True
        ),
        T = mo.ui.slider(start=300, stop=1000, step=10, value=600, show_value=True),
        xB0 = mo.ui.slider(start=0.10, stop=0.90, step=0.01, value=0.50, show_value=True)
    )
    return (setup,)


@app.cell
def _():
    8.314 * 673
    return


@app.cell
def _(np):
    ## definition of the free energy and the chemical potential

    kB = 1.380649e-23
    # T = 673.0

    nx = 128
    ny = 128
    dx = 2.0e-9
    dy = 2.0e-9

    Da = 1.0e-22
    Db = 5.0e-23

    # dt = 2.0e-2
    dt = (dx*dx/Da)*0.1
    print(dt)
    eps = 1.0e-8
    seed = 42


    def free_energy(x, omega_kbT_ratio=2.0, T=673.0):
        kbT = kB * T
        omega = omega_kbT_ratio * kbT
        x = np.clip(x, eps, 1.0 - eps)
        return omega * x * (1.0 - x) + kbT * (
            x * np.log(x) + (1.0 - x) * np.log(1.0 - x)
        )


    def chemical_potential(x, omega_kbT_ratio=2.0, T=673.0):
        kbT = kB * T
        omega = omega_kbT_ratio * kbT
        x = np.clip(x, eps, 1.0 - eps)
        return omega * (1.0 - 2.0 * x) + kbT * np.log(x / (1.0 - x))


    def dmu_dc(x, omega_kbT_ratio=2.0, T=673.0):
        kbT = kB * T
        omega = omega_kbT_ratio * kbT
        x = np.clip(x, eps, 1.0 - eps)
        return -2.0 * omega + kbT / (x * (1.0 - x))


    def darken_diffusivity(x, Da, Db):
        return Da * x + Db * (1.0 - x)


    def mobility(x, Da, Db, omega_kbT_ratio, T):
        return darken_diffusivity(x, Da, Db) / dmu_dc(x, omega_kbT_ratio, T)


    def dmdc(x, Da, Db, omega_kbT_ratio, T, h=1.0e-5):
        # Finite difference of dmu/dc
        xp = np.clip(x + h, 1.0e-8, 1.0 - 1.0e-8)
        xm = np.clip(x - h, 1.0e-8, 1.0 - 1.0e-8)
        return (
            mobility(xp, Da, Db, omega_kbT_ratio, T)
            - mobility(xm, Da, Db, omega_kbT_ratio, T)
        ) / (xp - xm)

    return (
        chemical_potential,
        dmdc,
        dmu_dc,
        dt,
        dx,
        dy,
        free_energy,
        kB,
        mobility,
        nx,
        ny,
        seed,
    )


@app.cell
def _(chemical_potential, dmu_dc, free_energy, kB, np, plt):
    def plot_energy(omega_kbT_ratio=4.5, T=673.0):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ax1, ax2 = axes[:]
        xB = np.linspace(0.01, 0.99, 200)
        g = free_energy(xB, omega_kbT_ratio, T)
        mu = chemical_potential(xB, omega_kbT_ratio, T)
        dmudc = dmu_dc(xB, omega_kbT_ratio, T)
        ax1.plot(xB, g / kB / T)
        ax1.set_xlabel("$x_B$")
        ax1.axhline(y=0, ls="--", color="grey")
    
        ax2.plot(xB, dmudc / kB / T)
        ax1.set_ylabel("$g^{{\\mathrm{{homo}}}}$ ($k_B T$)")
        ax2.set_xlabel("$x_B$")

        ax2.set_ylabel("$\\partial^2 g^{{\\mathrm{{homo}}}} / \\partial x_B$ ($k_B T$)")
        ax2.axhline(y=0, ls="--", color="grey")
        ax2.set_ylim(-10, 30)

    
        fig.tight_layout()
        return fig

    plot_energy()
    return


@app.cell
def _(dx, dy, np):
    def laplacian(c):
        # Calculate the laplacian field for c
        ce = np.roll(c, -1, axis=0)
        cw = np.roll(c,  1, axis=0)
        cn = np.roll(c, -1, axis=1)
        cs = np.roll(c,  1, axis=1)
        return (ce - 2.0 * c + cw) / dx**2 + (cn - 2.0 * c + cs) / dy**2

    def gradient_sq_term(c, mu):
        # Calculate the shifted gradient form
        ce = np.roll(c, -1, axis=0)
        cw = np.roll(c,  1, axis=0)
        cn = np.roll(c, -1, axis=1)
        cs = np.roll(c,  1, axis=1)

        mue = np.roll(mu, -1, axis=0)
        muw = np.roll(mu,  1, axis=0)
        mun = np.roll(mu, -1, axis=1)
        mus = np.roll(mu,  1, axis=1)

        dc2dx2 = ((ce - cw) * (mue - muw)) / (4.0 * dx**2)
        dc2dy2 = ((cn - cs) * (mun - mus)) / (4.0 * dy**2)
        return dc2dx2 + dc2dy2


    return gradient_sq_term, laplacian


@app.cell
def _():
    3.0e-14 * 1e18 / 5595.322
    return


@app.cell
def _(
    chemical_potential,
    dmdc,
    dt,
    gradient_sq_term,
    kB,
    laplacian,
    mobility,
    np,
    nx,
    ny,
    seed,
):
    rng = np.random.default_rng(seed)


    def run_simulation(
        nsteps=3000,
        xB0=0.5,
        noise=0.01,
        kappa_kbT_ratio=2.5,
        omega_kbT_ratio=2.5,
        T=673.0,
        Da=1e-11,
        Db=1e-13,
        every=600,
    ):
        x = xB0 + noise * rng.standard_normal((nx, ny))
        x = np.clip(x, 1.0e-8, 1.0 - 1.0e-8)

        kbT = kB * T
        omega = omega_kbT_ratio * kbT
        kappa = kappa_kbT_ratio * kbT
        frames = [x.copy()]

        for step in range(nsteps):
            mu_chem = chemical_potential(x, omega_kbT_ratio, T)
            mu = mu_chem - kappa * laplacian(x)

            M = mobility(x, Da, Db, omega_kbT_ratio, T)
            dMdc = dmdc(x, Da, Db, omega_kbT_ratio, T)

            nabla_mu = laplacian(mu)
            grad_term = gradient_sq_term(x, mu)

            dxdt = M * nabla_mu + dMdc * grad_term
            x = x + dt * dxdt
            x = np.clip(x, 1.0e-8, 1.0 - 1.0e-8)

            if (step + 1) % every == 0:
                frames.append(x.copy())

        return x, frames


    final_x, frames = run_simulation()
    return final_x, frames


@app.cell
def _(final_x):
    final_x
    return


@app.cell
def _(frames, plt):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(frames[1], cmap="bwr", vmin=0.0, vmax=1.0)
    ax.set_title("final composition field")
    fig.colorbar(im, ax=ax, label=r"$x_B$")
    plt.tight_layout()
    fig
    return


@app.cell
def _():

        # kappa = mo.ui.slider(
        #     start=0.1e-40, stop=8.0e-40, step=0.1e-40, value=2.0e-40,
        #     label=r"$\kappa$ [J m$^2$/site]"
        # )
        # xB0 = mo.ui.slider(
        #     start=0.05, stop=0.95, step=0.01, value=0.50,
        #     label=r"$x_{B0}$"
        # )
        # noise = mo.ui.slider(
        #     start=0.0, stop=0.05, step=0.001, value=0.01,
        #     label="initial noise"
        # )
        # nsteps = mo.ui.slider(
        #     start=100, stop=5000, step=100, value=1200,
        #     label="number of steps"
        # )
        # every = mo.ui.slider(
        #     start=5, stop=200, step=5, value=40,
        #     label="store every n steps"
        # )
    return


@app.cell
def _(setup):
    setup
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    from scipy.optimize import root

    return np, plt


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
