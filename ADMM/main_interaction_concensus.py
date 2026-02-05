"""
ADMM consensus pushing with repulsive cutoff force (FIXED to avoid deadlock)
---------------------------------------------------------------------------

Problem:
- Particle 1 (x) is actuated:  x_{t+1} = x_t + dt * u_t,  ||u_t||_inf <= u_max
- Particle 2 (y) is passive:   y_{t+1} = y_t + dt * f(x_t, y_t)

Goal:
- ONLY task cost: 0.5 * || y_T - y_goal ||^2
- Use ADMM with consensus x_t = z_t, where z_t is the "ghost" interaction center.

ADMM per MPC step:
  1) x-update: quadratic solve (fast) to match z - w with tiny control regularizer
  2) z-update: gradient steps on [0.5||y_T(z)-goal||^2 + (rho/2)||z-(x+w)||^2]
               using adjoint gradients through y dynamics with PLANNING force
  3) w-update: w <- w + (x - z)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# =============================================================================
# TRUE force (HARD cutoff) + Jacobians
# =============================================================================
def repulsive_force_and_jacobians_true(z, y, k=10.0, sigma=2.5, r_cut=2.0):
    """
    True repulsive "velocity" field pushing y away from z.
    Exactly zero for r >= r_cut (compact support).

    Returns:
      f:  (2,)
      Jy: df/dy (2x2)
      Jz: df/dz (2x2)
    """
    soft = 0.2
    d = y - z
    r = float(np.linalg.norm(d)) + 1e-12

    if r >= r_cut:
        f = np.zeros(2)
        J = np.zeros((2, 2))
        return f, J, J

    d_hat = d / r

    s = r / r_cut
    w = (1.0 - s * s) ** 2  # smooth to zero at r_cut

    gain = k * np.exp(-(r * r) / (sigma * sigma))
    mag = gain / (r + soft)
    A = w * mag

    f = A * d_hat

    # derivatives
    I = np.eye(2)
    outer = np.outer(d_hat, d_hat)

    dw_dr = -4.0 * s * (1.0 - s * s) / r_cut
    dmag_dr = mag * (-(2.0 * r) / (sigma * sigma) - 1.0 / (r + soft))
    dA_dr = dw_dr * mag + w * dmag_dr

    Jd = dA_dr * outer + (A / r) * (I - outer)
    Jy = Jd
    Jz = -Jd
    return f, Jy, Jz


# =============================================================================
# PLANNING force (tiny tail outside cutoff) + Jacobians
# =============================================================================
def repulsive_force_and_jacobians_planning(z, y, k=10.0, sigma=2.5, r_cut=2.0, tail=1e-2, tail_sigma=0.5):
    """
    Planning-only force:
      - Same behavior inside r_cut
      - Outside r_cut: tiny smooth tail so gradients don't vanish

    tail: amplitude scaling outside cutoff (e.g. 1e-2 or 1e-3)
    tail_sigma: decay length beyond cutoff
    """
    soft = 0.2
    d = y - z
    r = float(np.linalg.norm(d)) + 1e-12
    d_hat = d / r

    gain = k * np.exp(-(r * r) / (sigma * sigma))
    mag = gain / (r + soft)

    # inside-cutoff window
    s = r / r_cut
    w_in = (1.0 - s * s) ** 2 if r < r_cut else 0.0

    if r >= r_cut:
        # smooth tail (nonzero, tiny)
        # w = tail * exp(- (r-r_cut)^2 / tail_sigma^2 )
        dr = (r - r_cut)
        w = tail * np.exp(-(dr * dr) / (tail_sigma * tail_sigma))
        # dw/dr
        dw_dr = w * (-(2.0 * dr) / (tail_sigma * tail_sigma))
    else:
        w = w_in
        # dw/dr inside
        dw_dr = -4.0 * (r / r_cut) * (1.0 - (r / r_cut) ** 2) / r_cut

    A = w * mag
    f = A * d_hat

    # Jacobian wrt d
    I = np.eye(2)
    outer = np.outer(d_hat, d_hat)

    dmag_dr = mag * (-(2.0 * r) / (sigma * sigma) - 1.0 / (r + soft))
    dA_dr = dw_dr * mag + w * dmag_dr

    Jd = dA_dr * outer + (A / r) * (I - outer)
    Jy = Jd
    Jz = -Jd
    return f, Jy, Jz


# =============================================================================
# x-update: quadratic solve with dynamics (tridiagonal per dimension)
# =============================================================================
def solve_x_update(x0, z_traj, w_traj, dt, rho, eps_u, u_max):
    """
    Minimize:
      (rho/2) Σ_{t=0}^T ||x_t - z_t + w_t||^2  + (eps_u/2) Σ_{t=0}^{T-1} ||u_t||^2
    subject to:
      x_{t+1} = x_t + dt u_t
    with x0 fixed.

    Solve in x by eliminating u -> tridiagonal system.
    """
    T = z_traj.shape[0] - 1
    n = T  # vars x1..xT

    q = eps_u / (dt * dt)

    x_traj = np.zeros_like(z_traj)
    x_traj[0] = x0

    for dim in range(2):
        diag = np.zeros(n)
        off = np.zeros(n - 1)
        b = np.zeros(n)

        target = (z_traj[:, dim] - w_traj[:, dim])

        for i in range(n):
            t = i + 1

            # smoothness from u
            diag[i] += 2.0 * q if t < T else 1.0 * q
            if i > 0:
                off[i - 1] = -q
            if t == 1:
                b[i] += q * x0[dim]

            # consensus penalty
            diag[i] += rho
            b[i] += rho * target[t]

        A = np.zeros((n, n))
        np.fill_diagonal(A, diag)
        for i in range(n - 1):
            A[i, i + 1] = off[i]
            A[i + 1, i] = off[i]

        sol = np.linalg.solve(A, b)
        x_traj[1:, dim] = sol

    # enforce bounds by re-rolling
    u = (x_traj[1:] - x_traj[:-1]) / dt
    u = np.clip(u, -u_max, u_max)

    x_feas = np.zeros_like(x_traj)
    x_feas[0] = x0
    for t in range(T):
        x_feas[t + 1] = x_feas[t] + dt * u[t]

    return x_feas, u


# =============================================================================
# z-update: adjoint gradient descent through y dynamics (PLANNING force)
# =============================================================================
def rollout_y_and_adjoint_grad(z_traj, y0, y_goal, dt, rho, x_plus_w, k_force, sigma_force, r_cut, tail, tail_sigma):
    """
    J(z) = 0.5||y_T(z)-y_goal||^2 + (rho/2) Σ ||z_t - x_plus_w_t||^2

    Uses PLANNING force for rollout & Jacobians so gradients don't vanish.

    Returns:
      y_traj: (T+1,2)
      grad_z: (T+1,2)
    """
    T = z_traj.shape[0] - 1
    y = np.zeros((T + 1, 2))
    y[0] = y0

    Jy_list = []
    Jz_list = []

    # forward rollout
    for t in range(T):
        f, Jy, Jz = repulsive_force_and_jacobians_planning(z_traj[t], y[t], k=k_force, sigma=sigma_force, r_cut=r_cut, tail=tail, tail_sigma=tail_sigma)
        Jy_list.append(Jy)
        Jz_list.append(Jz)
        y[t + 1] = y[t] + dt * f

    # terminal adjoint
    lam = y[T] - y_goal

    grad_z = np.zeros_like(z_traj)

    # z_T only appears in consensus term
    grad_z[T] += rho * (z_traj[T] - x_plus_w[T])

    # backward pass
    for t in reversed(range(T)):
        grad_z[t] += rho * (z_traj[t] - x_plus_w[t])
        grad_z[t] += dt * (Jz_list[t].T @ lam)
        lam = (np.eye(2) + dt * Jy_list[t]).T @ lam

    return y, grad_z


def z_update_gradient(
    z_traj, x_traj, w_traj, y0, y_goal, dt, rho,
    k_force, sigma_force, r_cut,
    tail=1e-2, tail_sigma=0.5,
    z_gd_steps=25, z_lr=0.5
):
    x_plus_w = x_traj + w_traj
    z = z_traj.copy()

    for _ in range(z_gd_steps):
        _, grad = rollout_y_and_adjoint_grad(
            z, y0, y_goal, dt, rho, x_plus_w,
            k_force, sigma_force, r_cut, tail, tail_sigma
        )
        z = z - z_lr * grad

    return z


# =============================================================================
# MPC + ADMM loop
# =============================================================================
def run_sim_admm(
    x0=np.array([-2.0, 0.0]),
    y0=np.array([0.0, -2.0]),
    y_goal=np.array([2.5, 2.5]),
    dt=0.05,
    horizon_T=35,
    sim_steps=250,
    admm_iters=10,
    rho=2.0,
    eps_u=5e-3,
    u_max=2.0,
    # force
    k_force=10.0,
    sigma_force=2.5,
    r_cut=2.0,
    # planning tail (critical for avoiding deadlock)
    tail=1e-2,
    tail_sigma=0.5,
    # z optimizer
    z_gd_steps=25,
    z_lr=0.5,
    # slight warm-start bias toward current y (helps engage cutoff faster)
    warm_bias=0.3,
):
    x = np.asarray(x0, float).copy()
    y = np.asarray(y0, float).copy()
    y_goal = np.asarray(y_goal, float).copy()

    T = horizon_T

    # warm starts
    x_traj = np.tile(x, (T + 1, 1))
    z_traj = np.tile(x, (T + 1, 1))
    w_traj = np.zeros((T + 1, 2))

    xs = [x.copy()]
    ys = [y.copy()]

    for step in range(sim_steps):
        # warm-start bias: nudge z slightly toward y to help engage the interaction region
        z_traj = (1.0 - warm_bias) * z_traj + warm_bias * y

        for _ in range(admm_iters):
            # 1) x-update (fast closed-form)
            x_traj, u_traj = solve_x_update(x, z_traj, w_traj, dt, rho, eps_u, u_max)

            # 2) z-update (adjoint gradients, planning tail)
            z_traj = z_update_gradient(
                z_traj=z_traj,
                x_traj=x_traj,
                w_traj=w_traj,
                y0=y,
                y_goal=y_goal,
                dt=dt,
                rho=rho,
                k_force=k_force,
                sigma_force=sigma_force,
                r_cut=r_cut,
                tail=tail,
                tail_sigma=tail_sigma,
                z_gd_steps=z_gd_steps,
                z_lr=z_lr
            )

            # 3) dual update
            w_traj = w_traj + (x_traj - z_traj)

        # execute first control
        u0 = u_traj[0]
        x = x + dt * u0

        # TRUE y update uses HARD-cutoff physics
        f_true, _, _ = repulsive_force_and_jacobians_true(x, y, k=k_force, sigma=sigma_force, r_cut=r_cut)
        y = y + dt * f_true

        xs.append(x.copy())
        ys.append(y.copy())

        if step % 25 == 0:
            print("step", step, "||y-goal|| =", np.linalg.norm(y - y_goal), "| ||x-y|| =", np.linalg.norm(x - y))

        # shift warm starts
        x_traj[:-1] = x_traj[1:]
        x_traj[-1] = x_traj[-2]

        z_traj[:-1] = z_traj[1:]
        z_traj[-1] = z_traj[-2]

        w_traj[:-1] = w_traj[1:]
        w_traj[-1] = w_traj[-2]

    return np.array(xs), np.array(ys), y_goal


# =============================================================================
# GIF
# =============================================================================
def save_gif(xs, ys, goal, gif_path="admm_consensus_tail.gif", fps=30, trail=250, dpi=140):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    goal = np.asarray(goal)
    N = xs.shape[0]

    all_pts = np.vstack([xs, ys, goal[None, :]])
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    pad = 0.15 * np.maximum(mx - mn, 1e-3)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
    ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
    ax.grid(True)
    ax.set_title("ADMM consensus pushing (hard physics, soft planning tail)")

    ax.scatter([goal[0]], [goal[1]], marker="*", s=200, label="goal")
    ax.scatter([xs[0, 0]], [xs[0, 1]], marker="o", label="x start")
    ax.scatter([ys[0, 0]], [ys[0, 1]], marker="o", label="y start")

    x_trail_line, = ax.plot([], [], lw=2, label="x trail")
    y_trail_line, = ax.plot([], [], lw=2, label="y trail")
    x_dot, = ax.plot([], [], marker="o", markersize=8, linestyle="None", label="x(t)")
    y_dot, = ax.plot([], [], marker="o", markersize=8, linestyle="None", label="y(t)")
    link_line, = ax.plot([], [], lw=1, linestyle="--", label="link")
    ax.legend(loc="upper left")

    def init():
        x_trail_line.set_data([], [])
        y_trail_line.set_data([], [])
        x_dot.set_data([], [])
        y_dot.set_data([], [])
        link_line.set_data([], [])
        return x_trail_line, y_trail_line, x_dot, y_dot, link_line

    def update(i):
        j0 = max(0, i - trail)
        xseg = xs[j0:i + 1]
        yseg = ys[j0:i + 1]
        x_trail_line.set_data(xseg[:, 0], xseg[:, 1])
        y_trail_line.set_data(yseg[:, 0], yseg[:, 1])
        x_dot.set_data([xs[i, 0]], [xs[i, 1]])
        y_dot.set_data([ys[i, 0]], [ys[i, 1]])
        link_line.set_data([xs[i, 0], ys[i, 0]], [xs[i, 1], ys[i, 1]])
        return x_trail_line, y_trail_line, x_dot, y_dot, link_line

    ani = animation.FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=1000 / fps)
    writer = animation.PillowWriter(fps=fps)
    ani.save(gif_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved GIF to: {gif_path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    xs, ys, goal = run_sim_admm(
        horizon_T=35,
        sim_steps=500,
        admm_iters=10,
        rho=2.0,
        eps_u=5e-3,
        u_max=2.0,
        k_force=10.0,
        sigma_force=2.5,
        r_cut=2.0,
        tail=1e-2,
        tail_sigma=0.5,
        z_gd_steps=25,
        z_lr=0.5,
        warm_bias=0.3,
    )

    save_gif(xs, ys, goal, gif_path="results/admm_consensus_tail.gif", fps=30, trail=250, dpi=140)

    plt.figure()
    plt.plot(xs[:, 0], xs[:, 1], label="x path")
    plt.plot(ys[:, 0], ys[:, 1], label="y path")
    plt.scatter([goal[0]], [goal[1]], marker="*", s=200, label="goal")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("ADMM consensus pushing (single cost on y)")
    plt.show()
