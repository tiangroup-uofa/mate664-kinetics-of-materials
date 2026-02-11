import marimo

__generated_with = "0.19.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 2 Q2 Demo
    ## Kirkendall Effect Simulation
    This interactive app simulates the Kirkendall effect in a crystal frame (C-frame)
    using a probabilistic approach.

    - The system consists of blue and orange dots (atoms) and vacancies (grey).
    - At each step, a vacancy swaps with a neighboring atom (blue or orange) with user-defined probabilities.
    - You can adjust the number of atoms, vacancy concentration, and swap probabilities.
    - The Kirkendall effect should appear with our simulation when the C-frame boundaries are in exchange with atom-vacancy reservoirs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    setup = mo.md(
        """ ## Setup simulation environment
        Let's first setup a random simulation domain. 
     Please choose the following parameters:
 
     - Total sites x-direction each side $N_{{\\mathrm{{site}} }}$: {nsites}
     - Total rows $N_{{\\mathrm{{row}}}}$: {nrows}
     - Vacancy fraction $f_{{v}}$: {fvac}
     - Exchange prob. with Orange atoms $p_{{O}}$: {pO}
     - Exchange prob. with Blue atoms $p_{{B}}$: {pB}
     - Random generator seed: {seed}
 
     """
    ).batch(
        nsites=mo.ui.slider(5, 100, value=60, step=5, show_value=True),
        nrows=mo.ui.slider(5, 15, value=10, step=1, show_value=True),
        fvac=mo.ui.slider(0.01, 0.5, step=0.01, value=0.10, show_value=True),
        pO=mo.ui.slider(0.01, 1.0, step=0.05, value=0.75, show_value=True),
        pB=mo.ui.slider(0.01, 1.0, step=0.05, value=0.50, show_value=True),
        seed=mo.ui.slider(0, 255, value=42, show_value=True)
    )

    setup
    return (setup,)


@app.cell(hide_code=True)
def _(KirkendallSimulation, mo):
    ks = KirkendallSimulation.from_setup()
    mo.vstack([mo.md("### Initial setup is shown below:"),
    ks.render(show_count=True)])
    return (ks,)


@app.cell
def _(mo):
    # Button to run the simulation
    run_button = mo.ui.button(label="Run Simulation", on_click=lambda value: True)
    # UI components for simulation control
    num_steps = mo.ui.slider(
        start=100,
        stop=5000,
        value=500,
        step=100,
        label="Number of simulation steps",
        show_value=True,
        on_change=None,
    )
    mo.vstack(
        [
            mo.md("""
    ## Run KMC Simulation

    You can define how many steps to run such simulation. For saving time we use a "pseudo" kinetic Monte-Carlo method that all vacancy jumps are updated simultaneously in one step. In real KMC you want to run the simulation one vacancy at a time.
    """),
            num_steps,
            run_button,
        ]
    )
    return num_steps, run_button


@app.cell
def _(ks, mo, num_steps, plt, run_button):
    mo.stop(not run_button.value)

    ks.reset()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    nsteps = num_steps.value
    # print(nsteps)
    for i in mo.status.progress_bar(range(nsteps)):
        ks.step()

    ks.render(ax=ax, show_count=True)
    ax.set_title(f"Kirkendall Simulation after {nsteps} steps")
    return


@app.cell(hide_code=True)
def _(Axes, List, Optional, mcolors, np, plt, setup):
    class KirkendallSimulation:
        """Object-oriented implementation of the Kirkendall Effect simulation."""

        @classmethod
        def from_setup(cls):
            """Use the global setup"""
            values = setup.value
            return cls(
                width=values["nsites"] * 2,
                height=values["nrows"],
                vacancy_fraction=values["fvac"],
                p_orange=values["pO"],
                p_blue=values["pB"],
                seed=values["seed"],
            )

        def __init__(
            self,
            width: int,
            height: int,
            vacancy_fraction: float,
            p_orange: float,
            p_blue: float,
            seed: int,
        ):
            """
            Args:
                width: Width of the simulation domain
                height: Number of rows in 2D simulation
                vacancy_fraction: Fraction of vacancies in the system
                p_orange: Probability of vacancy swapping with orange atoms ("O")
                p_blue: Probability of vacancy swapping with blue atoms ("B")
                seed: RNG seed
            """
            self.width = int(width)
            self.height = int(height)
            self.vacancy_fraction = float(vacancy_fraction)
            self.p_orange = float(p_orange)
            self.p_blue = float(p_blue)
            self.seed = int(seed)

            self._generator = np.random.default_rng(seed=self.seed)

            # boundary escape bookkeeping (optional)
            self.escape_left = 0
            self.escape_right = 0

            self.reset()

        def _initialize_system(self) -> np.ndarray:
            H, W = self.height, self.width
            half_width = W // 2

            system = np.empty((H, W), dtype=object)

            # 1) left half O, right half B
            system[:, :half_width] = "O"
            system[:, half_width:] = "B"

            # add vacancies
            total_sites = H * W
            num_vacancies = int(total_sites * self.vacancy_fraction)

            flat_indices = self._generator.choice(
                total_sites, num_vacancies, replace=False
            )
            y = flat_indices // W
            x = flat_indices % W
            system[y, x] = "V"

            return system

        def _make_big(self) -> np.ndarray:
            """
            Build a padded lattice big with ghost rows/cols:

            - center: sys
            - x-ghost: left col = "O" reservoir, right col = "B" reservoir
            - y-ghost: wrap (periodic) by copying last/first row of sys
            """
            sys = self.system
            H, W = sys.shape
            big = np.empty((H + 2, W + 2), dtype=object)

            # center
            big[1:-1, 1:-1] = sys

            # x-ghost cols (reservoir)
            big[1:-1, 0] = "O"   # col -1
            big[1:-1, -1] = "B"  # col W

            # y-ghost rows (wrap / periodic)
            big[0, 1:-1] = sys[-1, :]   # row -1
            big[-1, 1:-1] = sys[0, :]   # row H

            # corners (consistent with x-reservoir)
            big[0, 0] = "O"
            big[0, -1] = "B"
            big[-1, 0] = "O"
            big[-1, -1] = "B"

            return big

        def reset(self):
            self.system = self._initialize_system()
            self.history: List[np.ndarray] = [self.system.copy()]
            self.time: List[int] = [0]

            self.escape_left = 0
            self.escape_right = 0
            return

        def step(self, debug_print=False) -> np.ndarray:
            sys = self.system
            H, W = sys.shape
            rng = self._generator

            # 1) record vacancy positions
            vy, vx = np.where(sys == "V")
            nV = vy.size
            if nV == 0:
                if debug_print:
                    print("Nv = 0")
                return sys

            # 2) propose neighbor moves (4-neighbor)
            dirs = rng.integers(0, 4, size=nV)
            dy = np.zeros(nV, dtype=int)
            dx = np.zeros(nV, dtype=int)
            dy[dirs == 0] = -1
            dy[dirs == 1] = +1
            dx[dirs == 2] = -1
            dx[dirs == 3] = +1

            ty_raw = vy + dy  # can be -1 or H
            tx_raw = vx + dx  # can be -1 or W

            # 3) build big lattice, map to big indices
            big = self._make_big()
            ty_big = ty_raw + 1  # [0..H+1]
            tx_big = tx_raw + 1  # [0..W+1]

            # neighbor species at proposed target
            nbr = big[ty_big, tx_big]

            # 4) accept probability from nbr species
            p = np.zeros(nV, dtype=float)
            p[nbr == "B"] = self.p_blue
            p[nbr == "O"] = self.p_orange
            # nbr == "V" => p=0 => auto reject

            accept_prob = rng.random(nV) < p
            if not np.any(accept_prob):
                if debug_print:
                    print("Nv =", int(np.sum(sys == "V")))
                return sys

            # 5) keep only unique target sites (random tie-break via shuffle)
            cand = np.nonzero(accept_prob)[0]
            # cand = cand[rng.permutation(cand.size)]
            # print(cand)

            # flat_t = ty_big[cand] * (W + 2) + tx_big[cand]
            # print(flat_t)
            # _, first_idx = np.unique(flat_t, return_index=True)
            # pick = cand[first_idx]
            pick = cand

            # 6) split picks: in-domain swap vs reservoir escape
            in_domain_x = (tx_big[pick] >= 1) & (tx_big[pick] <= W)
            to_left = (tx_big[pick] == 0)        # ghost col -1 (O reservoir)
            to_right = (tx_big[pick] == W + 1)   # ghost col W  (B reservoir)

            # 7a) in-domain swaps (y-wrap via ghost rows)
            if np.any(in_domain_x):
                sel = pick[in_domain_x]
                oy, ox = vy[sel], vx[sel]

                ny = (ty_big[sel] - 1) % H
                nx = tx_big[sel] - 1  # 0..W-1

                atom = sys[ny, nx].copy()
                sys[oy, ox] = atom
                sys[ny, nx] = "V"

            # 7b) reservoir escapes: fill origin with reservoir atom, count escapes
            n_left = 0
            n_right = 0

            if np.any(to_left):
                sel = pick[to_left]
                oy, ox = vy[sel], vx[sel]
                sys[oy, ox] = "O"  # keep swapped position as O
                n_left = sel.size

            if np.any(to_right):
                sel = pick[to_right]
                oy, ox = vy[sel], vx[sel]
                sys[oy, ox] = "B"  # keep swapped position as B
                n_right = sel.size

            self.escape_left += n_left
            self.escape_right += n_right
            #print(self.escape_left, self.escape_right)

            # 8) vacancy supply to keep vacancy count constant:
            # - left escape -> create vacancies on right boundary x=W-1
            # - right escape -> create vacancies on left boundary x=0
            if n_left > 0:
                col = W - 1
                candidates = np.nonzero(sys[:, col] != "V")[0]
                if candidates.size > 0:
                    k = min(n_left, candidates.size)
                    ypick = rng.choice(candidates, size=k, replace=False)
                    sys[ypick, col] = "V"

            if n_right > 0:
                col = 0
                candidates = np.nonzero(sys[:, col] != "V")[0]
                if candidates.size > 0:
                    k = min(n_right, candidates.size)
                    ypick = rng.choice(candidates, size=k, replace=False)
                    sys[ypick, col] = "V"

            # Dirty fix to add additional defects
            new_nV = int(np.sum(sys == "V"))
            if new_nV < nV:
                diff = nV - new_nV
                cany, canx = np.nonzero(sys != "V")
                if cand.size > 0:
                    k = min(diff, cand.size)
                    pick = rng.choice(range(len(cany)), size=k, replace=False)
                    for p_ in pick:
                        sys[cany[p_], canx[p_]] = "V"
            
            # # debug vacancy count
            if debug_print:
                print("Nv =", int(np.sum(sys == "V")))

            return sys

        def advance(self, nsteps: int, debug_print=False):
            """Advance multiple steps and store history/time."""
            for _ in range(int(nsteps)):
                self.step(debug_print=debug_print)
                self.history.append(self.system.copy())
                self.time.append(self.time[-1] + 1)

        def render(
            self,
            ax: Optional[Axes] = None,
            show_current: bool = True,
            index: int = -1,
            show_count: bool = False,
        ) -> Axes:
            """Render the current state of the simulation."""
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 6))

            tab_blue = np.array(mcolors.to_rgb("tab:blue"))
            tab_orange = np.array(mcolors.to_rgb("tab:orange"))
            tab_gray = np.array([0.7, 0.7, 0.7])
            color_map = {"B": tab_blue, "O": tab_orange, "V": tab_gray}

            _system = self.system if show_current else self.history[index]

            rgb_grid = np.zeros((_system.shape[0], _system.shape[1], 3))
            for i in range(_system.shape[0]):
                for j in range(_system.shape[1]):
                    rgb_grid[i, j] = color_map[_system[i, j]]

            ax.imshow(rgb_grid, origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])

            if show_count:
                blue_count = int(np.sum(self.system == "B"))
                orange_count = int(np.sum(self.system == "O"))
                vacancy_count = int(np.sum(self.system == "V"))

                stats_text = (
                    f"Blue: {blue_count}, Orange: {orange_count}, Vacancies: {vacancy_count}"
                )
                ax.text(
                    0.0,
                    -0.1,
                    stats_text,
                    transform=ax.transAxes,
                    color="white",
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.5),
                )

            return ax
    return (KirkendallSimulation,)


@app.cell(hide_code=True)
def _():
    # # Adjust p2 so that p1 + p2 = 100
    # p2_value = p2.value
    # p1_value = p1.value
    # print(p1_value, p2_value)
    # # System size
    # size = 2 * N.value
    # num_vacancies = int(size * vacancy_percent.value / 100)
    # num_blue = N.value
    # num_orange = N.value

    # # Initialize the system: left half blue, right half orange, then randomly assign vacancies
    # system = np.array(["B"] * num_blue + ["O"] * num_orange)

    # # Randomly assign vacancies
    # vacancy_indices = np.random.choice(size, num_vacancies, replace=False)
    # system[vacancy_indices] = "V"

    # system_init = system.copy()  # Save initial state for reset
    # system, system_init, p1_value, p2_value
    return


@app.cell(hide_code=True)
def _():
    # # UI for simulation steps
    # steps = mo.ui.slider(1, 5, value=3, label="Number of steps to simulate (x10e5)")
    # steps
    return


@app.cell(hide_code=True)
def _():
    # def simulate(
    #     system: np.ndarray, 
    #     steps: int, 
    #     pO: float, 
    #     pB: float, 
    #     p_fill: float = 1.0, 
    #     p_make: float = 1.0, 
    #     Nv_target: int = None, 
    #     kfb: float = 0.01,
    #     dims: int = 1,
    #     vac_update_fraction: tuple = (0.95, 1.0)
    # ) -> list:
    #     """
    #     Open bar with boundary reservoirs:
    #     - Interior: X<->V swaps with pB, pO
    #     - Left boundary exchanges with B reservoir
    #     - Right boundary exchanges with O reservoir
    #     - Optional feedback keeps Nv near Nv_target
    #     - Can operate in 1D or 2D (set dims=2 for 2D)

    #     Args:
    #         system: Initial system state array (1D or 2D)
    #         steps: Number of simulation steps
    #         pO: Probability of vacancy swapping with orange (%)
    #         pB: Probability of vacancy swapping with blue (%)
    #         p_fill: Probability to fill vacancy at boundary
    #         p_make: Probability to create vacancy at boundary
    #         Nv_target: Target number of vacancies (default: initial count)
    #         kfb: Feedback strength for vacancy control
    #         dims: Dimensionality (1 or 2)
    #         vac_update_fraction: Tuple of (min, max) fraction of vacancies to update per step

    #     Returns:
    #         List of system states throughout the simulation
    #     """
    #     system = system.copy()

    #     # Convert 1D array to 2D if dims=2 specified but 1D input provided
    #     if dims == 2 and system.ndim == 1:
    #         height = max(8, len(system) // 10)  # Default second dimension
    #         width = len(system)
    #         temp_system = np.full((height, width), "V", dtype=object)
    #         for i in range(height):
    #             temp_system[i, :] = system
    #         system = temp_system

    #     # Store dimensions for later use
    #     if dims == 1:
    #         N = len(system)
    #         history = [system.copy()]
    #     else:
    #         height, width = system.shape
    #         history = [system.copy()]

    #     pO /= 100.0
    #     pB /= 100.0

    #     if Nv_target is None:
    #         Nv_target = np.sum(system == "V")

    #     for _ in range(steps):
    #         Nv = np.sum(system == "V")
    #         # feedback factor: >1 means encourage vacancy creation, <1 discourage
    #         fb = np.exp(kfb * (Nv_target - Nv))

    #         # --- choose whether to do boundary event or interior event ---
    #         boundary_weight = 0.2
    #         if np.random.rand() < boundary_weight:
    #             # boundary event handling
    #             if dims == 1:
    #                 # 1D case - same as original
    #                 if np.random.rand() < 0.5:
    #                     # LEFT boundary coupled to B reservoir
    #                     if system[0] == "V":
    #                         if np.random.rand() < p_fill:
    #                             system[0] = "B"
    #                     elif system[0] == "B":
    #                         if np.random.rand() < p_make * fb:
    #                             system[0] = "V"
    #                 else:
    #                     # RIGHT boundary coupled to O reservoir
    #                     if system[N-1] == "V":
    #                         if np.random.rand() < p_fill:
    #                             system[N-1] = "O"
    #                     elif system[N-1] == "O":
    #                         if np.random.rand() < p_make * fb:
    #                             system[N-1] = "V"
    #             else:
    #                 # 2D case - randomly choose a boundary position
    #                 boundaries = ["left", "right", "top", "bottom"]
    #                 boundary = np.random.choice(boundaries)

    #                 if boundary == "left":
    #                     # Left boundary (Blue reservoir)
    #                     y = np.random.randint(0, height)
    #                     if system[y, 0] == "V":
    #                         if np.random.rand() < p_fill:
    #                             system[y, 0] = "B"
    #                     elif system[y, 0] == "B":
    #                         if np.random.rand() < p_make * fb:
    #                             system[y, 0] = "V"

    #                 elif boundary == "right":
    #                     # Right boundary (Orange reservoir)
    #                     y = np.random.randint(0, height)
    #                     if system[y, width-1] == "V":
    #                         if np.random.rand() < p_fill:
    #                             system[y, width-1] = "O"
    #                     elif system[y, width-1] == "O":
    #                         if np.random.rand() < p_make * fb:
    #                             system[y, width-1] = "V"

    #                 elif boundary == "top":
    #                     # Top boundary (mixed reservoir based on x-position)
    #                     x = np.random.randint(0, width)
    #                     # Top half is coupled to Blue, bottom half to Orange
    #                     if x < width // 2:
    #                         if system[0, x] == "V":
    #                             if np.random.rand() < p_fill:
    #                                 system[0, x] = "B"
    #                         elif system[0, x] == "B":
    #                             if np.random.rand() < p_make * fb:
    #                                 system[0, x] = "V"
    #                     else:
    #                         if system[0, x] == "V":
    #                             if np.random.rand() < p_fill:
    #                                 system[0, x] = "O"
    #                         elif system[0, x] == "O":
    #                             if np.random.rand() < p_make * fb:
    #                                 system[0, x] = "V"

    #                 elif boundary == "bottom":
    #                     # Bottom boundary (mixed reservoir based on x-position)
    #                     x = np.random.randint(0, width)
    #                     # Top half is coupled to Blue, bottom half to Orange
    #                     if x < width // 2:
    #                         if system[height-1, x] == "V":
    #                             if np.random.rand() < p_fill:
    #                                 system[height-1, x] = "B"
    #                         elif system[height-1, x] == "B":
    #                             if np.random.rand() < p_make * fb:
    #                                 system[height-1, x] = "V"
    #                     else:
    #                         if system[height-1, x] == "V":
    #                             if np.random.rand() < p_fill:
    #                                 system[height-1, x] = "O"
    #                         elif system[height-1, x] == "O":
    #                             if np.random.rand() < p_make * fb:
    #                                 system[height-1, x] = "V"
    #         else:
    #             # interior vacancy swap - update multiple vacancies
    #             if dims == 1:
    #                 # 1D case - get all vacancy positions
    #                 v_positions = np.where(system == "V")[0]
    #                 if len(v_positions) == 0:
    #                     history.append(system.copy())
    #                     continue

    #                 # Determine number of vacancies to update this step
    #                 min_updates = max(1, int(len(v_positions) * vac_update_fraction[0]))
    #                 max_updates = max(1, int(len(v_positions) * vac_update_fraction[1]))
    #                 num_updates = np.random.randint(min_updates, max_updates + 1)
    #                 num_updates = min(num_updates, len(v_positions))

    #                 # Randomly select vacancies to update
    #                 update_indices = np.random.choice(
    #                     len(v_positions), num_updates, replace=False
    #                 )

    #                 # Process each selected vacancy
    #                 for idx in update_indices:
    #                     v = v_positions[idx]
    #                     direction = np.random.choice([-1, 1])
    #                     n = v + direction

    #                     # reflecting ends: vacancy can't hop out
    #                     if n < 0 or n >= N:
    #                         continue

    #                     neighbor = system[n]
    #                     if neighbor == "B":
    #                         swap_prob = pB
    #                     elif neighbor == "O":
    #                         swap_prob = pO
    #                     else:
    #                         swap_prob = 0.0

    #                     if np.random.rand() < swap_prob:
    #                         system[v], system[n] = system[n], system[v]
    #             else:
    #                 # 2D case - get all vacancy positions
    #                 v_positions = np.where(system == "V")
    #                 if len(v_positions[0]) == 0:
    #                     history.append(system.copy())
    #                     continue

    #                 # Determine number of vacancies to update this step
    #                 min_updates = max(1, int(len(v_positions[0]) * vac_update_fraction[0]))
    #                 max_updates = max(1, int(len(v_positions[0]) * vac_update_fraction[1]))
    #                 num_updates = np.random.randint(min_updates, max_updates + 1)
    #                 num_updates = min(num_updates, len(v_positions[0]))

    #                 # Randomly select vacancies to update
    #                 update_indices = np.random.choice(
    #                     len(v_positions[0]), num_updates, replace=False
    #                 )

    #                 # Process each selected vacancy
    #                 for idx in update_indices:
    #                     vy, vx = v_positions[0][idx], v_positions[1][idx]

    #                     # Choose random direction (up, down, left, right)
    #                     directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    #                     dy, dx = directions[np.random.randint(0, 4)]
    #                     ny, nx = vy + dy, vx + dx

    #                     # Check boundaries
    #                     if ny < 0 or ny >= height or nx < 0 or nx >= width:
    #                         continue

    #                     neighbor = system[ny, nx]
    #                     if neighbor == "B":
    #                         swap_prob = pB
    #                     elif neighbor == "O":
    #                         swap_prob = pO
    #                     else:
    #                         swap_prob = 0.0

    #                     if np.random.rand() < swap_prob:
    #                         system[vy, vx], system[ny, nx] = system[ny, nx], system[vy, vx]

    #         history.append(system.copy())

    #     return history
    return


@app.cell(hide_code=True)
def _():
    # def plot_kirkendall_simulation(
    #     system: np.ndarray, 
    #     steps: int,
    #     p1_value: float, 
    #     p2_value: float,
    #     filename: str = "kirkendall_simulation.mp4",
    #     dims: int = 2
    # ) -> plt.Figure:
    #     """
    #     Create an MP4 video of the Kirkendall effect simulation and save it to file.

    #     Args:
    #         system: The initial system state
    #         steps: Number of simulation steps
    #         p1_value: Probability of vacancy swapping with orange (%)
    #         p2_value: Probability of vacancy swapping with blue (%)
    #         filename: Path where to save the MP4 file (default: "kirkendall_simulation.mp4")
    #         dims: Dimensionality (1 or 2)

    #     Returns:
    #         The matplotlib figure with the rendered simulation
    #     """
    #     from matplotlib.animation import FFMpegWriter
    #     import matplotlib.animation as animation
    #     from IPython.display import Video, display

    #     # Run the simulation
    #     history = simulate(system, int(steps * 1e3), p1_value, p2_value, dims=dims)

    #     colors = {"B": "blue", "O": "orange", "V": "gray"}
    #     color_map = {"B": [0, 0, 1], "O": [1, 0.5, 0], "V": [0.7, 0.7, 0.7]}

    #     # Number of frames to use (sample from history to keep video length reasonable)
    #     num_frames = min(100, len(history))
    #     frame_indices = np.linspace(0, len(history)-1, num_frames, dtype=int)

    #     # Set up the writer
    #     writer = FFMpegWriter(fps=10)

    #     if dims == 1:
    #         # 1D case - scatter plot
    #         fig, ax = plt.subplots(figsize=(10, 2))
    #         ax.set_xlim(-1, len(system))
    #         ax.set_ylim(-0.5, 0.5)
    #         ax.set_yticks([])
    #         ax.set_title("Kirkendall Effect Simulation (1D)")
    #         ax.set_xlabel("Position")

    #         with writer.saving(fig, filename, dpi=100):
    #             for frame_idx in range(num_frames):
    #                 # Clear the axis for new frame
    #                 ax.clear()
    #                 ax.set_xlim(-1, len(system))
    #                 ax.set_ylim(-0.5, 0.5)
    #                 ax.set_yticks([])
    #                 ax.set_title("Kirkendall Effect Simulation (1D)")
    #                 ax.set_xlabel("Position")

    #                 # Get the state at this frame
    #                 idx = frame_indices[frame_idx]
    #                 dots = history[idx]

    #                 # Create scatter plot for this frame
    #                 ax.scatter(
    #                     np.arange(len(dots)), 
    #                     np.zeros(len(dots)),
    #                     c=[colors[dot] for dot in dots],
    #                     s=100
    #                 )

    #                 # Add frame counter
    #                 ax.text(0.02, 0.95, f'Step: {idx}', transform=ax.transAxes)

    #                 # Grab the frame and save
    #                 writer.grab_frame()

    #         plt.close(fig)

    #         # Create a new figure to return (showing the final state)
    #         final_fig, final_ax = plt.subplots(figsize=(10, 2))
    #         final_ax.set_xlim(-1, len(system))
    #         final_ax.set_ylim(-0.5, 0.5)
    #         final_ax.set_yticks([])
    #         final_ax.set_title(
    #             f"Kirkendall Effect Simulation (1D Final State at Step {len(history)-1})"
    #         )
    #         final_ax.set_xlabel("Position")

    #         # Display final state
    #         final_dots = history[-1]
    #         final_ax.scatter(
    #             np.arange(len(final_dots)),
    #             np.zeros(len(final_dots)),
    #             c=[colors[dot] for dot in final_dots],
    #             s=100
    #         )

    #     else:
    #         # 2D case - use imshow
    #         height, width = history[0].shape
    #         fig, ax = plt.subplots(figsize=(10, 10 * height / width))
    #         ax.set_title("Kirkendall Effect Simulation (2D)")

    #         # Convert string grid to RGB values for visualization
    #         def grid_to_rgb(grid):
    #             rgb_grid = np.zeros((grid.shape[0], grid.shape[1], 3))
    #             for i in range(grid.shape[0]):
    #                 for j in range(grid.shape[1]):
    #                     rgb_grid[i, j] = color_map[grid[i, j]]
    #             return rgb_grid

    #         with writer.saving(fig, filename, dpi=100):
    #             for frame_idx in range(num_frames):
    #                 # Clear the axis for new frame
    #                 ax.clear()
    #                 ax.set_title("Kirkendall Effect Simulation (2D)")

    #                 # Get the state at this frame
    #                 idx = frame_indices[frame_idx]
    #                 grid = history[idx]

    #                 # Create image from grid
    #                 rgb_grid = grid_to_rgb(grid)
    #                 ax.imshow(rgb_grid)
    #                 ax.set_xticks([])
    #                 ax.set_yticks([])

    #                 # Add frame counter
    #                 ax.text(0.02, 0.95, f'Step: {idx}', transform=ax.transAxes, 
    #                        color='white', fontweight='bold')

    #                 # Grab the frame and save
    #                 writer.grab_frame()

    #         plt.close(fig)

    #         # Create a new figure to return (showing the final state)
    #         final_fig, final_ax = plt.subplots(figsize=(10, 10 * height / width))
    #         final_ax.set_title(
    #             f"Kirkendall Effect Simulation (2D Final State at Step {len(history)-1})"
    #         )

    #         # Display final state
    #         final_grid = history[-1]
    #         final_rgb = grid_to_rgb(final_grid)
    #         final_ax.imshow(final_rgb)
    #         final_ax.set_xticks([])
    #         final_ax.set_yticks([])

    #     # Display the video
    #     print(f"Animation saved to: {filename}")
    #     video = Video(filename, embed=True, html_attributes="controls loop autoplay")
    #     display(video)

    #     plt.tight_layout()
    #     return final_fig

    # # Create the animation
    # fig = plot_kirkendall_simulation(system, steps.value, p1_value, p2_value)
    # plt.gca()
    return


@app.function(hide_code=True)
def method(cls):
    def deco(fn):
        setattr(cls, fn.__name__, fn)
        return fn
    return deco


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from typing import List, Tuple, Dict, Optional, Union, Any
    import matplotlib.colors as mcolors
    return Axes, List, Optional, mcolors, np, plt


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
