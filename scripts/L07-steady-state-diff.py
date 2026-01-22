import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(
    c1_slider,
    c2_slider,
    length_slider,
    mo,
    plot_concentration,
    plot_selector,
):
    _ax, _fluxes = plot_concentration()
    _flux_md = mo.md(
        """
    Flux at point 1 (abs. unit):

    - $J(Cart)={cart_flux}$ 
    - $J(Cylinder)={cylinder_flux}$
    - $J(Sphere)={sphere_flux}$""".format(
            cart_flux=f"{_fluxes[0]:.2e}" if _fluxes[0] is not None else "NaN",
            cylinder_flux=f"{_fluxes[1]:.2e}" if _fluxes[1] is not None else "NaN",
            sphere_flux=f"{_fluxes[2]:.2e}" if _fluxes[2] is not None else "NaN",
        )
    )
    mo.vstack(
        [
            mo.hstack([length_slider, c1_slider, c2_slider], widths=[1, 1, 1]),
            mo.hstack([plot_selector, _flux_md], widths=[1, 2]),
            mo.hstack([_ax, mo.image(src="./public/L07-scheme.svg")], widths=[1, 1])
        ]
    )
    return


@app.cell
def _(mo):
    # UI components
    length_slider = mo.ui.slider(start=2, stop=20, step=1, value=5, show_value=True, label="$L$ (abs. unit)")

    c1_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=1.0, show_value=True, label="$c_1$ (abs. unit)")

    c2_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.0, show_value=True, label="$c_2$ (abs. unit)")

    plot_selector = mo.ui.array(
        label="Which solution to plot?",
        elements=[
        mo.ui.switch(value=False, label="1D Cartesian"),
        mo.ui.switch(value=False, label="Cylindrical"),
        mo.ui.switch(value=False, label="Spherical"),
    ])


    return c1_slider, c2_slider, length_slider, plot_selector


@app.cell
def _(
    c1_slider,
    c2_slider,
    c_profile_cart,
    c_profile_cylinder,
    c_profile_sphere,
    length_slider,
    plot_selector,
    plt,
):
    def plot_concentration() -> tuple:
        """
        Creates a visualization of concentration profiles with boundary conditions.
    
        Uses slider values to determine plot parameters:
        - length_slider: determines end position
        - c1_slider, c2_slider: boundary concentrations at start and end
        - plot_selector: determines which profiles to plot (cartesian/cylinder/sphere)
    
        Returns:
            tuple: (matplotlib axis object, list of flux values for each geometry)
        """
        # --- Initialize parameters ---
        start = 1
        end = length_slider.value
        c1 = c1_slider.value
        c2 = c2_slider.value
        _plot_cart, _plot_cylinder, _plot_sphere = plot_selector.value
    
        # --- Create and configure plot ---
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_xlim(0, end + 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Position (abs. unit)")
        ax.set_ylabel("Concentration (abs. unit)")
    
        # --- Plot boundaries and boundary conditions ---
        ax.axvline(start, ls="--", color="grey")
        ax.axvline(end, ls="--", color="grey")
        ax.axvspan(0, start, color="gray", alpha=0.5)
        ax.axvspan(end, end + 1, color="gray", alpha=0.5)
        ax.scatter(start, c1, color='C5', s=100, zorder=5)
        ax.scatter(end, c2, color='C5', s=100, zorder=5)
    
        # --- Plot concentration profiles ---
        fluxes = []
        if _plot_cart:
            _x, _c = c_profile_cart(c1, c2, start, end)
            ax.plot(_x, _c, color="C1", label="1D Cart.")
            # Calculate flux as negative gradient at first point
            flux = -1 * (_c[1] - _c[0]) / (_x[1] - _x[0])
            fluxes.append(flux)
        else:
            fluxes.append(None)

        if _plot_cylinder:
            _x, _c = c_profile_cylinder(c1, c2, start, end)
            ax.plot(_x, _c, color="C2", label="Cylinder")
            # Calculate flux as negative gradient at first point
            flux = -1 * (_c[1] - _c[0]) / (_x[1] - _x[0])
            fluxes.append(flux)
        else:
            fluxes.append(None)

        if _plot_sphere:
            _x, _c = c_profile_sphere(c1, c2, start, end)
            ax.plot(_x, _c, color="C3", label="Sphere")
            # Calculate flux as negative gradient at first point
            flux = -1 * (_c[1] - _c[0]) / (_x[1] - _x[0])
            fluxes.append(flux)
        else:
            fluxes.append(None)

        ax.legend()
        return ax, fluxes
    return (plot_concentration,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    def c_profile_cart(c1: float, c2: float, x1: float, x2: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate concentration profile for Cartesian coordinates.
    
        Args:
            c1: Concentration at position x1
            c2: Concentration at position x2
            x1: Starting position
            x2: Ending position
    
        Returns:
            Tuple containing (position array, concentration array)
        """
        x_range = np.linspace(x1, x2, 200)
        c = c1 - (c1 - c2) * (x_range - x1)/(x2 - x1)
        return x_range, c

    def c_profile_cylinder(c1: float, c2: float, r1: float, r2: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate concentration profile for cylindrical coordinates.
    
        Args:
            c1: Concentration at radius r1
            c2: Concentration at radius r2
            r1: Inner radius
            r2: Outer radius
    
        Returns:
            Tuple containing (radial position array, concentration array)
        """
        x_range = np.linspace(r1, r2, 200)
        c = c1 - (c1 - c2) * np.log(x_range/r1) / np.log(r2/r1)
        return x_range, c

    def c_profile_sphere(c1: float, c2: float, r1: float, r2: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate concentration profile for spherical coordinates.
    
        Args:
            c1: Concentration at radius r1
            c2: Concentration at radius r2
            r1: Inner radius
            r2: Outer radius
    
        Returns:
            Tuple containing (radial position array, concentration array)
        """
        x_range = np.linspace(r1, r2, 200)
        c = c1 + (c2 - c1) * (1/x_range - 1/r1) / (1/r2 - 1/r1)
        return x_range, c
    return c_profile_cart, c_profile_cylinder, c_profile_sphere, plt


if __name__ == "__main__":
    app.run()
