"""
Toy 2D ADMM-style “pushing” simulation (repulsive interaction)

Fixes:
- Repulsive physics
- Push-consistent z update
- NEW: projection to keep x behind y relative to the goal direction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# ----------------------------
# Repulsive interaction model
# ----------------------------
def interaction_force_repulsive(
    x: np.ndarray,
    y: np.ndarray,
    k: float = 10.0,
    sigma: float = 2.5,
    r_cut: float = 2.0,
) -> np.ndarray:
    """
    Repulsive force pushing y away from x, ACTIVE ONLY for r < r_cut.

    Uses a smooth compact-support window w(r) that:
      - w = 1 at r=0
      - w -> 0 smoothly as r -> r_cut
      - w = 0 for r >= r_cut
    """
    soft = 0.2
    d = y - x
    r = float(np.linalg.norm(d)) + 1e-12
    if r >= r_cut:
        return np.zeros_like(y)

    d_hat = d / r

    # Base magnitude (short-range gain + softening)
    gain = k * np.exp(-(r * r) / (sigma * sigma))
    mag = gain / (r + soft)

    # Smooth compact-support window: w(r) = (1 - (r/r_cut)^2)^2 for r < r_cut, else 0
    s = r / r_cut
    w = (1.0 - s * s) ** 2

    return (w * mag) * d_hat

def rollout_y_repulsive(y0: np.ndarray, z_traj: np.ndarray, dt: float, k: float, sigma: float, r_cut: float) -> np.ndarray:
    T = z_traj.shape[0] - 1
    y = np.zeros_like(z_traj)
    y[0] = y0
    for t in range(T):
        F = interaction_force_repulsive(z_traj[t], y[t], k=k, sigma=sigma, r_cut=r_cut)
        y[t + 1] = y[t] + dt * F
    return y


def push_target(y: np.ndarray, y_goal: np.ndarray, d_push: float = 1.0) -> np.ndarray:
    v = y_goal - y
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return y.copy()
    g_hat = v / n
    return y - d_push * g_hat


def push_target_traj(y_traj: np.ndarray, y_goal: np.ndarray, d_push: float) -> np.ndarray:
    z_des = np.zeros_like(y_traj)
    for i in range(y_traj.shape[0]):
        z_des[i] = push_target(y_traj[i], y_goal, d_push=d_push)
    return z_des


# ----------------------------
# NEW: "behind-y" projection
# ----------------------------
def project_behind_y(x: np.ndarray, y: np.ndarray, y_goal: np.ndarray, margin: float = 0.2) -> np.ndarray:
    """
    Enforce (x - y)·g_hat <= -margin. If violated, project x onto the plane behind y.
    """
    v = y_goal - y
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return x
    g_hat = v / n

    s = float(np.dot(x - y, g_hat))  # positive => x is in front
    if s > -margin:
        # shift x backward along g_hat so that dot becomes exactly -margin
        x = x - (s + margin) * g_hat
    return x


def project_traj_behind_y(x_traj: np.ndarray, y: np.ndarray, y_goal: np.ndarray, margin: float = 0.2) -> np.ndarray:
    """
    Apply the same behind-y constraint to every x_t using the CURRENT y (toy but effective).
    """
    out = x_traj.copy()
    for t in range(out.shape[0]):
        out[t] = project_behind_y(out[t], y, y_goal, margin=margin)
    return out


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
    T = z_traj.shape[0] - 1
    if T < 1:
        raise ValueError("horizon_T must be >= 1")

    n = T
    q = lam_u / (dt * dt)

    x_traj = np.zeros((T + 1, 2), dtype=float)
    x_traj[0] = x0

    for dim in range(2):
        diag = np.zeros(n, dtype=float)
        off = np.zeros(n - 1, dtype=float)
        b = np.zeros(n, dtype=float)

        a = (z_traj[:, dim] - w_traj[:, dim])
        g = float(y_goal[dim])
        rref = float(y_ref[dim])

        for i in range(n):
            t = i + 1

            diag[i] += 2.0 * q if t < T else 1.0 * q
            diag[i] += lam_x + lam_xy + rho

            b[i] += lam_x * g
            b[i] += lam_xy * rref
            b[i] += rho * a[t]

            if i > 0:
                off[i - 1] = -q

            if t == 1:
                b[i] += q * float(x0[dim])

        A = np.zeros((n, n), dtype=float)
        np.fill_diagonal(A, diag)
        for i in range(n - 1):
            A[i, i + 1] = off[i]
            A[i + 1, i] = off[i]

        sol = np.linalg.solve(A, b)
        x_traj[1:, dim] = sol

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
    sim_steps=400,
    admm_iters=15,
    # costs
    lam_u=0.3,
    lam_x=0.0,
    lam_xy=10.0,    # stronger push-position tracking
    rho=0.8,        # reduce consensus domination
    # bounds
    u_max=2.0,
    # repulsive interaction
    k_force=10.0,
    sigma_force=2.5,
    r_cut=2.0,
    # z update heuristic strength
    kappa=0.6,
    kappa_power=2.0,
    # pushing geometry
    d_push=1.2,
    behind_margin=0.3,
):
    x = np.asarray(x0, dtype=float).copy()
    y = np.asarray(y0, dtype=float).copy()
    y_goal = np.asarray(y_goal, dtype=float).copy()

    x_traj = np.tile(x, (horizon_T + 1, 1))
    z_traj = np.tile(x, (horizon_T + 1, 1))
    w_traj = np.zeros((horizon_T + 1, 2), dtype=float)

    xs = [x.copy()]
    ys = [y.copy()]

    for step in range(sim_steps):
        for _ in range(admm_iters):
            y_ref = push_target(y, y_goal, d_push=d_push)

            # 1) x-update
            x_traj, u_traj = solve_x_trajectory_quadratic(
                x0=x,
                z_traj=z_traj,
                w_traj=w_traj,
                y_goal=y_goal,
                y_ref=y_ref,
                dt=dt,
                lam_u=lam_u,
                lam_x=lam_x,
                lam_xy=lam_xy,
                rho=rho,
                u_max=u_max,
            )

            # NEW: ensure x trajectory stays behind y
            x_traj = project_traj_behind_y(x_traj, y, y_goal, margin=behind_margin)

            # 2) z-update (push-consistent)
            z_bar = x_traj + w_traj
            y_bar = rollout_y_repulsive(y0=y, z_traj=z_bar, dt=dt, k=k_force, sigma=sigma_force, r_cut=r_cut)
            z_des = push_target_traj(y_bar, y_goal, d_push=d_push)

            weights = (np.linspace(0.0, 1.0, horizon_T + 1) ** kappa_power)[:, None]
            z_traj = (1.0 - kappa * weights) * z_bar + (kappa * weights) * z_des

            # 3) dual update
            w_traj = w_traj + (x_traj - z_traj)

        # Execute MPC action
        u0 = u_traj[0]
        x = x + dt * u0

        # NEW: enforce behind constraint at executed state too
        x = project_behind_y(x, y, y_goal, margin=behind_margin)

        # True y update
        F_true = interaction_force_repulsive(x, y, k=k_force, sigma=sigma_force, r_cut=r_cut)
        y = y + dt * F_true

        if step % 20 == 0:
            g_hat = (y_goal - y) / (np.linalg.norm(y_goal - y) + 1e-12)
            progress = float(np.dot(F_true, g_hat))
            print("step", step, "progress along goal dir =", progress)

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
def save_gif(xs, ys, goal, gif_path="admm_pushing.gif", fps=30, trail=200, dpi=120):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    goal = np.asarray(goal)

    N = xs.shape[0]
    if N < 2:
        raise ValueError(f"Need at least 2 frames to make a GIF, got N={N}.")
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
    ax.set_title("ADMM-style pushing (x repels y toward goal)")

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

    writer = animation.PillowWriter(fps=fps)
    ani.save(gif_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved GIF to: {gif_path}")


if __name__ == "__main__":
    xs, ys, goal = run_sim()
    save_gif(xs, ys, goal, gif_path="admm_pushing.gif", fps=30, trail=250, dpi=140)

    plt.figure()
    plt.plot(xs[:, 0], xs[:, 1], label="Particle 1 (actuated) path")
    plt.plot(ys[:, 0], ys[:, 1], label="Particle 2 (passive) path")
    plt.scatter([xs[0, 0]], [xs[0, 1]], marker="o", label="x start")
    plt.scatter([ys[0, 0]], [ys[0, 1]], marker="o", label="y start")
    plt.scatter([goal[0]], [goal[1]], marker="*", s=200, label="goal (y*)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("ADMM-style pushing: x repels y toward goal")
    plt.show()
