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
    return (micropip,)


@app.cell
async def _(micropip):
    await micropip.install("plotly")
    import plotly
    import numpy as np
    import plotly.graph_objects as go
    return go, np


@app.cell
def _(go, np):
    def plot(log10_D: float, log10_vm: float):
        D_AB: float = 10 ** (log10_D)     # m^2/s
        v_m: float  = 10 ** (log10_vm)    # m/s
    
        # Time horizon (seconds)
        t_end: float = 120.0
        t = np.linspace(0, t_end, 200)
    
        # Length scales
        L_diff = 6.0 * np.sqrt(D_AB * t)   # m
        L_conv = v_m * t                   # m

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=L_diff, mode="lines", 
                                 name="Diffusion: $6\\sqrt{D_{AB}t}$"))
        fig.add_trace(go.Scatter(x=t, y=L_conv, mode="lines", 
                                 name="Convection: $v_m t$"))

        fig.update_layout(
            title=f"Length Scale vs Time (0–{t_end:.0f} s)",
            xaxis_title="Time, t (s)",
            yaxis_title="Length, L (m)",
            yaxis_type="log",
            hovermode="x unified",
            height=680,
            template="plotly_white",
            font=dict(size=26),
            legend=dict(font=dict(size=22)),
        )
        return fig

    plot(log10_D=-10, log10_vm=-3)
    return


if __name__ == "__main__":
    app.run()
