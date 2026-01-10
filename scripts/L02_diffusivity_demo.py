import marimo

__generated_with = "0.19.0"
app = marimo.App(
    width="full",
)


@app.cell
def _():
    import marimo as mo
    return

@app.cell
def _():
    import micropip
    return

@app.cell
def _():
    await micropip.install("plotly")
    import plotly

@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go

    # Constants
    _R_gas_constant: float = 8.314  # J/(mol·K)
    _ref_temp: float = 298.15  # K (25 °C)

    # Reference diffusivities at _ref_temp (m^2/s)
    _D_gas_ref: float = 1e-5
    _D_liquid_ref: float = 5e-10
    _D_solid_ref: float = 1e-14

    # Temperature dependency parameters
    # For gases: D ~ T^exponent
    _gas_exponent: float = 1.75
    # For liquids/solids: Arrhenius-like D ~ exp(-Ea/RT)
    _Ea_liquid: float = 15_000  # J/mol (typical for diffusion in water)
    _Ea_solid: float = 50_000  # J/mol (typical for diffusion in solids)

    # Temperature slider


    def calculate_diffusivity_gas(temp: float) -> float:
        """Calculate gas diffusivity based on temperature."""
        return _D_gas_ref * (temp / _ref_temp)**_gas_exponent

    def calculate_diffusivity_liquid(temp: float) -> float:
        """Calculate liquid diffusivity based on temperature."""
        return _D_liquid_ref * np.exp(
            (_Ea_liquid / _R_gas_constant) * (1 / _ref_temp - 1 / temp)
        )

    def calculate_diffusivity_solid(temp: float) -> float:
        """Calculate solid diffusivity based on temperature."""
        return _D_solid_ref * np.exp(
            (_Ea_solid / _R_gas_constant) * (1 / _ref_temp - 1 / temp)
        )

    # Calculate diffusivities at the current slider temperature
    # _D_gas_current: float = calculate_diffusivity_gas(current_temp.value)
    # _D_liquid_current: float = calculate_diffusivity_liquid(current_temp.value)
    # _D_solid_current: float = calculate_diffusivity_solid(current_temp.value)

    # Display current values
    # _display_values = mo.md(
    #     f"""
    #     ### Diffusivity at {current_temp.value:.2f} K
    #     ({current_temp.value - 273.15:.2f} °C)
    #     - **Gas:** `{_D_gas_current:.2e}` m²/s
    #     - **Liquid:** `{_D_liquid_current:.2e}` m²/s
    #     - **Solid:** `{_D_solid_current:.2e}` m²/s
    #     """
    # )

    # Generate data for plotting over the temperature range
    _temps_for_plot: np.ndarray = np.linspace(273.15, 373.15, 100)
    _D_gas_plot: np.ndarray = np.array(
        [calculate_diffusivity_gas(t) for t in _temps_for_plot]
    )
    _D_liquid_plot: np.ndarray = np.array(
        [calculate_diffusivity_liquid(t) for t in _temps_for_plot]
    )
    _D_solid_plot: np.ndarray = np.array(
        [calculate_diffusivity_solid(t) for t in _temps_for_plot]
    )

    # Create Plotly figure
    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=_temps_for_plot,
        y=_D_gas_plot,
        mode='lines',
        name='Gas',
        line=dict(color='blue'),
        hovertemplate=(
            'Temperature: %{x:.2f} K<br>'
            'Diffusivity: %{y:.2e} m²/s<extra>Gas</extra>'
        )
    ))
    _fig.add_trace(go.Scatter(
        x=_temps_for_plot,
        y=_D_liquid_plot,
        mode='lines',
        name='Liquid',
        line=dict(color='red'),
        hovertemplate=(
            'Temperature: %{x:.2f} K<br>'
            'Diffusivity: %{y:.2e} m²/s<extra>Liquid</extra>'
        )
    ))
    _fig.add_trace(go.Scatter(
        x=_temps_for_plot,
        y=_D_solid_plot,
        mode='lines',
        name='Solid',
        line=dict(color='green'),
        hovertemplate=(
            'Temperature: %{x:.2f} K<br>'
            'Diffusivity: %{y:.2e} m²/s<extra>Solid</extra>'
        )
    ))

    _fig.update_layout(
        title="Diffusivity vs. Temperature for Different Phases",
        xaxis_title="Temperature (K)",
        yaxis_title="Diffusivity (m²/s)",
        yaxis_type="log",  # Use log scale for y-axis due to large range
        hovermode="x unified",
        legend_title="Phase",
        width=1200,
        template="plotly_white",
        font=dict(
            size=26  
        ),
    )
    D_AB_plot = _fig
    return (D_AB_plot,)


@app.cell
def _(D_AB_plot):
    D_AB_plot
    return


if __name__ == "__main__":
    app.run()
