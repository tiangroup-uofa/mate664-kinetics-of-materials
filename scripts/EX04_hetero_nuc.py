import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo, np):
    setup = mo.md("""
    ## Simulating multistage heterogeneous nucleation in a pit

    For a pit with side length $l$ (arb. unit) and solid angle $\\alpha$, 
    we can calcualte the total free energy of nucleation $\\Delta G$ as

    $$
    \\Delta G = V(n)* G + \\Delta A(n) * \\gamma
    $$

    where $G$ and $\\gamma$ are scaled free energy and surface energy, respectively.

    How will the free energy profile look like when we change the change the volume of cone? We can plot the normalized
    $\\Delta G / \\Delta G_{{c, \\text{{homo}}}}$ as a measure with respect to **homogeneous critical barrier**. 
    For simplification
    we will monitor the total volume of the nucleation normalized by the **homogeneous critical barrier**, that is $n / n_c$

    Please change the following:

    - Length $l$ (arb. unit): {l0} 
    - Solid angle $\\alpha$ (°): {alpha} 
    - Ratio $\\dfrac{{S}}{{G}}$ (arb. unit): {k} 
    - Location on the curve $n/n_c$: {location}

    """).batch(
        l0=mo.ui.slider(steps=np.linspace(0.1, 5, 32), show_value=True),
        alpha=mo.ui.slider(steps=np.arange(15, 151, 5), value=60, show_value=True),
        k=mo.ui.slider(
            steps=np.arange(0.5, 1.5, 0.05), value=1.00, show_value=True
        ),
        location=mo.ui.slider(
            steps=np.linspace(0.0, 2.0, 32), value=0.0, show_value=True
        ),
    )
    setup
    return (setup,)


@app.cell(hide_code=True)
def _(cos, np, plot_single, plt, radians, setup, sin):
    def plot_geometry(ax, l0_val, l1_val, alpha_val, l2_val, alpha2_val):
        """
        Plots the geometric representation of the nucleation stages.

        Args:
            l0_val (float): The base length of the initial cone.
            alpha_val (float): The apex angle of the initial cone in degrees.
            l1_val (float): The effective length for stage 1.
            l2_val (float): The effective length for stage 2 and 3.
            alpha2_val (float): The apex angle for the cap in degrees.

        Returns:
            matplotlib.axes.Axes: The axes object containing the plot.
        """
        theta_cone = radians(alpha_val / 2)
        r_cone = l0_val * sin(theta_cone)
        h_cone = l0_val * cos(theta_cone)

        # Base of the cone
    
        ax.plot([-r_cone, 0, r_cone], [h_cone, 0, h_cone], color='grey', linestyle='-')

        # Stage 1: Partial cone
        if l1_val < l0_val:
            theta_arc_stage1 = radians(alpha_val / 2)
            theta_vals = np.linspace(-theta_arc_stage1, theta_arc_stage1, 100)
            x_arc = l1_val * np.sin(theta_vals)
            y_arc = l1_val * np.cos(theta_vals)

            ax.plot(x_arc, y_arc, color="grey", linestyle='-')
            ax.fill_between(x_arc, y_arc, np.abs(x_arc) / np.tan(theta_cone), color="tab:blue", alpha=0.3)

        # Stage 2: Spherical cap with a varying angle
        elif alpha2_val < 180:
            theta_arc_stage2 = radians(alpha2_val / 2)
            theta_vals = np.linspace(-theta_arc_stage2, theta_arc_stage2, 100)
            x_arc = l2_val * np.sin(theta_vals)
            y_arc = l2_val * (np.cos(theta_vals) - np.cos(theta_arc_stage2)) + h_cone

            ax.plot(x_arc, y_arc, color="grey", linestyle='-')
            ax.fill_between(x_arc, y_arc, np.abs(x_arc) / np.tan(theta_cone), color="tab:blue", alpha=0.3)

        # Stage 3: Full spherical cap
        else:
            theta_vals = np.linspace(-np.pi / 2, np.pi / 2, 100)
            x_arc = l2_val * np.sin(theta_vals)
            y_arc = l2_val * (np.cos(theta_vals)) + h_cone

            ax.plot(x_arc, y_arc, color="grey", linestyle='-')
            ax.fill_between(x_arc, y_arc, h_cone, color="tab:blue", alpha=0.3)
            ax.fill_between([-r_cone, 0, r_cone], [h_cone, 0, h_cone], [h_cone, ] * 3, color="tab:blue", alpha=0.3)

        ax.set_aspect('equal', adjustable='box')
        return ax

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1 = axes[0]
    ax2 = axes[1]
    ax1, param = plot_single(
        ax1,
        l0=setup.value["l0"],
        alpha=setup.value["alpha"],
        show_location=setup.value["location"],
        k_value=setup.value["k"],
    )

    # Extract parameters for drawing
    l0_val = setup.value["l0"]
    l1_val, alpha_val, l2_val, alpha2_val = param

    # Plot the geometry
    ax2 = plot_geometry(ax2, l0_val, *param)
    ax2.set_title("Nucleus Geometry")

    fig
    return


@app.cell(hide_code=True)
def _(ideal_cone, ideal_plane, np, profile, vol_cone):
    def plot_single(ax, l0, alpha=60, show_location=0, k_value=0.5):
        """Compare the profiles with perfect flat plane and infinite cone"""
        # Pit-system
        steps = 128
        v, dG, params = profile(l0=l0, steps=steps, alpha=alpha, k=k_value)
        v_p, dG_p = ideal_plane(l0=l0, alpha=alpha, steps=steps, alpha2=180, k=k_value)
        v_b, dG_b = ideal_plane(l0=0, alpha2=360, steps=steps, k=k_value)
        v_cone, dG_cone = ideal_cone(L_max=5, alpha=alpha, steps=steps, k=k_value)
        nc = v_b[np.argmax(dG_p)]
        gc_bulk = np.max(dG_b)

        v_cone_max = vol_cone(l0, alpha)

        ax.plot(v[:steps] / nc, dG[:steps] / gc_bulk, label="Stage 1")
        ax.plot(v[steps: 2 * steps] / nc,
                dG[steps: 2 * steps] / gc_bulk, label="Stage 2")
        ax.plot(v[steps * 2: steps * 3] / nc,
                dG[steps * 2: steps * 3] / gc_bulk, label="Stage 3")

        # Bulk
        ax.plot(v_b / nc, dG_b / gc_bulk, "-.", alpha=0.6,
                # label="Homogeneous nuceation"
        )
    
        # Planar system
        ax.plot(v_p / nc, dG_p / gc_bulk, "--", alpha=0.6,
                # label="Ideal flat surface"
        )

        # cone
        ax.plot(v_cone / nc, dG_cone / gc_bulk, "--", alpha=0.6,
                # label="Infinite cone"
        )
    
        ax.axhline(y=0, color="grey", alpha=0.3)
        ax.set_xlim(0, 2)
        ax.set_ylim(-1, 1)
        ax.set_title("Volume of cone: {:.2f} $n_{{\\mathrm{{c}}}}$".
                     format(v_cone_max / nc))
        ax.set_ylabel(r"$\Delta G / \Delta G_{\mathrm{c, homo}}$")
        ax.set_xlabel(r"$n / n_{\mathrm{c, homo}}$")

        # Display the current location on the free energy curve
        if show_location >= 0:
            # Interpolate to find the dG value at the exact show_location
            # We use the 'v' and 'dG' arrays, normalized by 'nc' and 'gc_bulk' respectively
            normalized_v = v / nc
            normalized_dG = dG / gc_bulk
        
            # Use numpy.interp to find the dG value at show_location
            # Ensure that show_location is within the range of normalized_v
            if show_location >= normalized_v.min() and show_location <= normalized_v.max():
                interpolated_dG = np.interp(show_location, normalized_v, normalized_dG)
                ax.plot(show_location, interpolated_dG, 'ko', markersize=5)  # 'ko' for black circle
            else:
                pass

            # Obtain the current parameter
            idx = np.abs(normalized_v - show_location).argmin()
            param = params[idx]
        
        
                # If show_location is outside the plotted range, find the closest point
                # This part is similar to the original logic, as it handles out-of-bounds cases
                # 
                # ax.plot(normalized_v[idx], normalized_dG[idx], 'ko', markersize=5)

        ax.legend(loc="lower right", frameon=True, fontsize="smaller")
        return ax, param

    return (plot_single,)


@app.cell(hide_code=True)
def _(area_cap, np, vol_cap, vol_cone):
    def ideal_plane(l0=0, r_max=3,
                    alpha=60,
                    alpha2=180,
                    steps=128,
                    k=0.7):
        """
        Calculates the Delta Phi - V profile of an ideally flat plane.

        This function simulates an ideal flat plane by varying the radius of a spherical cap
        while keeping the cone's length and angle constant.

        Args:
            l0 (float): The length of the cone. Defaults to 0, implying no cone.
            r_max (float): The maximum radius of the spherical cap. Defaults to 3.
            alpha (float): The apex angle of the cone in degrees. Defaults to 60.
            alpha2 (float): The apex angle of the cone defining the spherical cap in degrees.
                            Defaults to 180, representing a full sphere.
            steps (int): The number of steps to discretize the radius variation. Defaults to 128.
            k (float): A proportionality constant used in the Delta Phi calculation. Defaults to 0.7.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - v (np.ndarray): An array of calculated volumes.
                - delta (np.ndarray): An array of Delta Phi values (Delta Phi = -V + k*S).
        """
        # l0 = 0 no delay
        params = [(l0, alpha, r, alpha2) for r in np.linspace(0, r_max, steps)]
        v = np.array([vol_cone(l, a) + vol_cap(r, a2)
                      for l, a, r, a2 in params])
        s = np.array([area_cap(r, a2)
                      for l, a, r, a2 in params])
        delta = -v + k * s
        return v, delta


    def ideal_cone(L_max=5,
                   alpha=60,
                   steps=128,
                   k=0.7):
        """
        Calculates the Delta Phi - V profile of an ideal, infinitely large cone.

        This function simulates an ideal cone by varying its length while keeping the
        apex angle constant and assuming no cap.

        Args:
            L_max (float): The maximum length of the cone. Defaults to 5.
            alpha (float): The apex angle of the cone in degrees. Defaults to 60.
            steps (int): The number of steps to discretize the length variation. Defaults to 128.
            k (float): A proportionality constant used in the Delta Phi calculation. Defaults to 0.7.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - v (np.ndarray): An array of calculated volumes.
                - delta (np.ndarray): An array of Delta Phi values (Delta Phi = -V + k*S).
        """
        # l0 = 0 no delay
        params = [(l, alpha, 0, 0) for l in np.linspace(0, L_max, steps)]
        v = np.array([vol_cone(l, a) + vol_cap(l, a)
                      for l, a, r, a2 in params])
        s = np.array([area_cap(l, a)
                      for l, a, r, a2 in params])
        delta = -v + k * s
        return v, delta

    return ideal_cone, ideal_plane


@app.cell(hide_code=True)
def _(area_cap, np, radians, sin, vol_cap, vol_cone):
    def profile(l0=2, dr=5, alpha=60, steps=128,
                k=0.7):
        """
        Calculates the Delta Phi - V profile of a mixed system.

        The profile is generated by varying the geometry of a cone and a cap.
        The process is divided into three steps:
        1. Both cone and cap dimensions vary: cone (l, alpha), cap (l, alpha) where l <= l0.
        2. Cone dimension is fixed, cap dimension varies: cone (l0, alpha), cap (r, alpha') where alpha' is varied from alpha/2 to 90 degrees.
        3. Cone dimension is fixed, cap radius varies: cone (l0, alpha), cap (r, 180 degrees) where r varies from the maximum radius in step 2 up to l0*sin(alpha/2) + dr.

        Args:
            l0 (float): Initial length of the cone.
            dr (float): Additional radius for the cap in the third step.
            alpha (float): The apex angle of the cone in degrees.
            steps (int): The number of steps to use for each part of the profile generation.
            k (float): A proportionality constant.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - v (np.ndarray): Array of calculated volumes.
                - delta (np.ndarray): Array of calculated Delta Phi values.
        """
        theta = alpha / 2
        r_max = l0 * sin(radians(theta)) + dr
        params = []
        # Step 1
        ll = np.linspace(0, l0, steps)
        for l in ll:
            params.append((l, alpha, l, alpha))
        # Step 2
        tt = np.linspace(theta, 90, steps)
        for t in tt:
            params.append((l0, alpha,
                           l0 * sin(radians(theta))
                           / sin(radians(t)),
                           2 * t))
        # Step 3
        rr = np.linspace(l0 * sin(radians(theta)), r_max, steps)
        [params.append((l0, alpha, r, 180)) for r in rr]

        res = []
        for p in params:
            l, alpha, r, alpha2 = p
            v = vol_cone(l, alpha) + vol_cap(r, alpha2)
            s = area_cap(r, alpha2)
            res.append((v, s))

        res = np.array(res)
        v = res[:, 0]
        s = res[:, 1]
        delta = -v + k * s
        return v, delta, params

    return (profile,)


@app.cell(hide_code=True)
def _(cos, pi, radians, sin):
    def vol_cone(l, alpha):
        """
        Calculates the volume of a cone.

        Args:
            l (float): The slant height of the cone.
            alpha (float): The apex angle of the cone in degrees.

        Returns:
            float: The volume of the cone.
        """
        theta = radians(alpha / 2)
        r = l * sin(theta)
        h = l * cos(theta)
        v = 1.0 / 3 * pi * r * r * h
        return v


    def vol_cap(r, phi):
        """
        Calculates the volume of a spherical cap.

        Args:
            r (float): The radius of the sphere from which the cap is cut.
            phi (float): The apex angle of the cone that defines the cap's height in degrees.

        Returns:
            float: The volume of the spherical cap.
        """
        theta = radians(phi / 2)
        v = 1.0 / 3 * pi * r ** 3 * (2 + cos(theta)) * (1 - cos(theta)) ** 2
        return v


    def area_cap(r, phi):
        """
        Calculates the surface area of a spherical cap.

        Args:
            r (float): The radius of the sphere from which the cap is cut.
            phi (float): The apex angle of the cone that defines the cap's height in degrees.

        Returns:
            float: The surface area of the spherical cap.
        """
        theta = radians(phi / 2)
        s = 2 * pi * (1 - cos(theta)) * r ** 2
        return s

    return area_cap, vol_cap, vol_cone


@app.cell(hide_code=True)
def _():
    import numpy as np
    from numpy import pi, cos, sin, radians
    import matplotlib.pyplot as plt

    return cos, np, pi, plt, radians, sin


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
