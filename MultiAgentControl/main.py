"""
Toy 2D ADMM-style “shepherding” simulation

- Particle 1 (x) is actuated (we choose control u).
- Particle 2 (y) is unactuated; it moves via a short-range attractive force toward a “ghost” trajectory z.

ADMM split (educational / toy):
    x,u update: quadratic trajectory optimization pulling x toward (z - w) and toward goal.
    z update: heuristic “best response” that nudges z ahead of y toward the goal while staying close to (x + w).
    w update: scaled dual update enforcing consensus x ≈ z.

This is intentionally simple and stable (good for learning + tinkering).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# ----------------------------
# Dynamics: short-range attraction
# ----------------------------
def gamma(r: float, k: float = 3.0, sigma: float = 1.0) -> float:
    """Smooth, bounded gain: strong only when close."""
    return k * np.exp(-(r * r) / (sigma * sigma))


def interaction_force(attractor: np.ndarray, y: np.ndarray, k: float = 3.0, sigma: float = 1.0) -> np.ndarray:
    """
    Force pulling y toward attractor (e.g., z or x).
    F = gamma(||d||) * d, where d = attractor - y.
    """
    d = attractor - y
    r = float(np.linalg.norm(d)) + 1e-12
    return gamma(r, k=k, sigma=sigma) * d


def rollout_y(y0: np.ndarray, z_traj: np.ndarray, dt: float, k: float = 3.0, sigma: float = 1.0) -> np.ndarray:
    """
    Roll out y forward given a fixed z trajectory:
        y_{t+1} = y_t + dt * F(z_t, y_t)
    """
    T = z_traj.shape[0] - 1
    y = np.zeros_like(z_traj)
    y[0] = y0
    for t in range(T):
        F = interaction_force(z_traj[t], y[t], k=k, sigma=sigma)
        y[t + 1] = y[t] + dt * F
    return y


# ----------------------------
# Quadratic x-trajectory solve (dense tridiagonal per dimension)
# ----------------------------
def solve_x_trajectory_quadratic(
    x0: np.ndarray,
    z_traj: np.ndarray,
    w_traj: np.ndarray,
    y_goal: np.ndarray,
    y_ref: np.ndarray,
    dt: float,
    lam_u: float,
    lam_x: float,
    lam_xy: float,
    rho: float,
    u_max: float,
):
    """
    Solve for x_{0:T} minimizing:
        sum_{t=0}^{T-1} (lam_u/2)||u_t||^2
      + sum_{t=0}^{T}   (lam_x/2)||x_t - y_goal||^2
      + sum_{t=0}^{T}   (rho/2)  ||x_t - z_t + w_t||^2
    subject to:
        x_{t+1} = x_t + dt u_t

    Eliminate u_t = (x_{t+1}-x_t)/dt -> quadratic in x with tridiagonal structure.
    Then compute u, clip to bounds, and re-roll x forward to enforce feasibility.
    """
    T = z_traj.shape[0] - 1
    n = T  # unknowns are x1..xT (x0 fixed)

    if T < 1:
        raise ValueError("horizon_T must be >= 1")

    q = lam_u / (dt * dt)  # coefficient on (x_{t+1}-x_t)^2

    x_traj = np.zeros((T + 1, 2), dtype=float)
    x_traj[0] = x0

    for dim in range(2):
        diag = np.zeros(n, dtype=float)
        off = np.zeros(n - 1, dtype=float)
        b = np.zeros(n, dtype=float)

        a = (z_traj[:, dim] - w_traj[:, dim])  # target for x from augmented term
        g = y_goal[dim]

        for i in range(n):  # i corresponds to time t = i+1
            t = i + 1

            # Smoothness term contributions
            # For x_t:
            # - if 1 <= t <= T-1: appears in (x_t - x_{t-1})^2 and (x_{t+1}-x_t)^2 -> 2q
            # - if t == T: appears only in (x_T - x_{T-1})^2 -> q
            diag[i] += 2.0 * q if t < T else 1.0 * q

            # State + augmented penalties
            diag[i] += lam_x + lam_xy + rho

            # RHS: lam_x * goal + rho * (z - w)
            b[i] += lam_x * g + lam_xy * y_ref[dim]
            b[i] += rho * a[t]

            # Off-diagonals from smoothness coupling with previous/next
            if i > 0:
                off[i - 1] = -q

            # Contribution of fixed x0 from (x1 - x0)^2
            if t == 1:
                b[i] += q * x0[dim]

        # Build dense tridiagonal (fine for toy sizes)
        A = np.zeros((n, n), dtype=float)
        np.fill_diagonal(A, diag)
        for i in range(n - 1):
            A[i, i + 1] = off[i]
            A[i + 1, i] = off[i]

        sol = np.linalg.solve(A, b)
        x_traj[1:, dim] = sol

    # Recover u, clip, and re-roll for feasibility
    u = (x_traj[1:] - x_traj[:-1]) / dt
    u = np.clip(u, -u_max, u_max)

    x_feas = np.zeros_like(x_traj)
    x_feas[0] = x0
    for t in range(T):
        x_feas[t + 1] = x_feas[t] + dt * u[t]

    return x_feas, u


# ----------------------------
# ADMM-style MPC loop
# ----------------------------
def run_sim(
    x0=np.array([-2.0, 0.0]),
    y0=np.array([0.0, -2.0]),
    y_goal=np.array([2.5, 2.5]),
    dt=0.05,
    horizon_T=25,
    sim_steps=140,
    admm_iters=15,
    # costs
    lam_u=0.3,
    lam_x=0.5,
    rho=2.0,
    # bounds
    u_max=2.0,
    # interaction
    k_force=8.0,
    sigma_force=3.0,
    # z update heuristic strength
    kappa=0.25,
    kappa_power=2.0,
):
    """
    Each simulation step:
      - run admm_iters of ADMM on a horizon of length horizon_T (MPC),
      - apply first control u0 to x,
      - update y one step using true interaction with x (not z),
      - shift warm-start variables.
    """

    x = np.asarray(x0, dtype=float).copy()
    y = np.asarray(y0, dtype=float).copy()
    y_goal = np.asarray(y_goal, dtype=float).copy()

    x_traj = np.tile(x, (horizon_T + 1, 1))
    z_traj = np.tile(x, (horizon_T + 1, 1))
    w_traj = np.zeros((horizon_T + 1, 2), dtype=float)

    xs = [x.copy()]
    ys = [y.copy()]

    for _step in range(sim_steps):
        # ADMM iterations on horizon
        for _ in range(admm_iters):
            # 1) x-update
            x_traj, u_traj = solve_x_trajectory_quadratic(
                x0=x,
                z_traj=z_traj,
                w_traj=w_traj,
                y_goal=y_goal,
                y_ref=y,          
                dt=dt,
                lam_u=lam_u,
                lam_x=lam_x,
                lam_xy=2.0,      
                rho=rho,
                u_max=u_max,
            )

            # 2) z-update heuristic
            z_bar = x_traj + w_traj
            y_bar = rollout_y(y0=y, z_traj=z_bar, dt=dt, k=k_force, sigma=sigma_force)

            weights = (np.linspace(0.0, 1.0, horizon_T + 1) ** kappa_power)[:, None]
            z_traj = z_bar + kappa * weights * (y_goal - y_bar)

            # 3) dual update (scaled form)
            w_traj = w_traj + (x_traj - z_traj)

        # Execute MPC action (first control)
        u0 = u_traj[0]
        x = x + dt * u0

        # True y update uses interaction with actual x
        F_true = interaction_force(x, y, k=k_force, sigma=sigma_force)
        if _step % 20 == 0:
            print("||F_true|| =", np.linalg.norm(F_true))
        y = y + dt * F_true

        xs.append(x.copy())
        ys.append(y.copy())

        # Shift warm starts
        x_traj[:-1] = x_traj[1:]
        x_traj[-1] = x_traj[-2]

        z_traj[:-1] = z_traj[1:]
        z_traj[-1] = z_traj[-2]

        w_traj[:-1] = w_traj[1:]
        w_traj[-1] = w_traj[-2]

    return np.array(xs), np.array(ys), y_goal


# ----------------------------
# Animation helper (GIF)
# ----------------------------
def save_gif(xs, ys, goal, gif_path="admm_shepherding.gif", fps=30, trail=200, dpi=120):
    """
    Create an animation of the trajectories and save as a GIF.
    Requires Pillow: pip install pillow
    """
    N = xs.shape[0]
    if N < 2:
        raise ValueError(f"Need at least 2 frames to make a GIF, got N={N}.")

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    goal = np.asarray(goal)

    N = xs.shape[0]
    if ys.shape[0] != N:
        raise ValueError("xs and ys must have the same length")

    all_pts = np.vstack([xs, ys, goal[None, :]])
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    pad = 0.15 * np.maximum(mx - mn, 1e-3)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
    ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
    ax.grid(True)
    ax.set_title("ADMM-style shepherding (x actuated → y to goal)")

    ax.scatter([goal[0]], [goal[1]], marker="*", s=200, label="goal (y*)")
    ax.scatter([xs[0, 0]], [xs[0, 1]], marker="o", label="x start")
    ax.scatter([ys[0, 0]], [ys[0, 1]], marker="o", label="y start")

    x_trail_line, = ax.plot([], [], lw=2, label="x trail")
    y_trail_line, = ax.plot([], [], lw=2, label="y trail")
    x_dot, = ax.plot([], [], marker="o", markersize=8, linestyle="None", label="x(t)")
    y_dot, = ax.plot([], [], marker="o", markersize=8, linestyle="None", label="y(t)")
    link_line, = ax.plot([], [], lw=1, linestyle="--", label="link")
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    ax.legend(loc="upper left")

    def init():
        x_trail_line.set_data([], [])
        y_trail_line.set_data([], [])
        x_dot.set_data([], [])
        y_dot.set_data([], [])
        link_line.set_data([], [])
        time_text.set_text("")
        return x_trail_line, y_trail_line, x_dot, y_dot, link_line, time_text

    def update(i):
        j0 = max(0, i - trail)
        xseg = xs[j0:i + 1]
        yseg = ys[j0:i + 1]

        x_trail_line.set_data(xseg[:, 0], xseg[:, 1])
        y_trail_line.set_data(yseg[:, 0], yseg[:, 1])

        x_dot.set_data([xs[i, 0]], [xs[i, 1]])
        y_dot.set_data([ys[i, 0]], [ys[i, 1]])

        link_line.set_data([xs[i, 0], ys[i, 0]], [xs[i, 1], ys[i, 1]])

        time_text.set_text(f"t = {i}")
        return x_trail_line, y_trail_line, x_dot, y_dot, link_line, time_text


    ani = animation.FuncAnimation(
        fig, update, frames=N, init_func=init, blit=True, interval=1000 / fps
    )

    try:
        writer = animation.PillowWriter(fps=fps)
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(
            "Failed to create PillowWriter. Install Pillow with: pip install pillow"
        ) from e

    ani.save(gif_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved GIF to: {gif_path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    xs, ys, goal = run_sim()
    save_gif(xs, ys, goal, gif_path="admm_shepherding.gif", fps=30, trail=250, dpi=140)

    # Optional static plot
    plt.figure()
    plt.plot(xs[:, 0], xs[:, 1], label="Particle 1 (actuated) path")
    plt.plot(ys[:, 0], ys[:, 1], label="Particle 2 (passive) path")
    plt.scatter([xs[0, 0]], [xs[0, 1]], marker="o", label="x start")
    plt.scatter([ys[0, 0]], [ys[0, 1]], marker="o", label="y start")
    plt.scatter([goal[0]], [goal[1]], marker="*", s=200, label="goal (y*)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("ADMM-style shepherding: actuated x moves passive y to goal")
    plt.show()
