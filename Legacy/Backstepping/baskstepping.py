import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# -----------------------------
# Backstepping controller + sim
# -----------------------------
def backstepping_control(p1, p2, g, k1=2.0, k2=2.0):
    """
    Two-point backstepping for:
        p1_dot = u
        p2_dot = p1 - p2
    Goal: p2 -> g
    """
    e1 = p2 - g                      # object tracking error
    alpha = p2 - k1 * e1             # virtual control for p1
    z = p1 - alpha                   # backstepping error

    # dynamics pieces needed for alpha_dot
    p2_dot = (p1 - p2)
    e1_dot = p2_dot                     # g is constant
    alpha_dot = p2_dot - k1 * e1_dot    # = (1-k1)(p1-p2)

    # backstepping control
    u = alpha_dot - k2 * z - e1
    return u

def dynamics(x, g, k1, k2):
    """
    State x = [p1x, p1y, p2x, p2y]
    """
    p1 = x[0:2]
    p2 = x[2:4]
    u = backstepping_control(p1, p2, g, k1=k1, k2=k2)
    p1_dot = u
    p2_dot = p1 - p2
    return np.hstack([p1_dot, p2_dot])

def rk4_step(f, x, dt, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ---------------
# Main parameters
# ---------------
dt = 0.01
T = 3.0
N = int(T / dt)

g = np.array([2.0, 1.0])         # desired goal for point 2
p1_0 = np.array([-2.0, 2.0])     # initial point 1
p2_0 = np.array([-2.0, -2.0])    # initial point 2
x0 = np.hstack([p1_0, p2_0])

k1_gain = 2.0
k2_gain = 3.0

# -------------
# Simulate
# -------------
traj = np.zeros((N + 1, 4))
traj[0] = x0
x = x0.copy()

for i in range(N):
    x = rk4_step(dynamics, x, dt, g, k1_gain, k2_gain)
    traj[i + 1] = x

p1_traj = traj[:, 0:2]
p2_traj = traj[:, 2:4]
time = np.linspace(0.0, T, N + 1)

# -----------------------------
# Animation 
# -----------------------------
# Decimate frames for speed/size
stride = 5
frame_ids = np.arange(0, N + 1, stride)

# Plot bounds
all_pts = np.vstack([p1_traj, p2_traj, g[None, :]])
pad = 0.5
xmin, ymin = all_pts.min(axis=0) - pad
xmax, ymax = all_pts.max(axis=0) + pad

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title("Two-point backstepping: p2 → goal by controlling p1")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.grid(True)

# Static goal marker
ax.scatter([g[0]], [g[1]], marker="x", s=120, label="goal g")

# Trail lines (paths)
p1_line, = ax.plot([], [], lw=2, label="p1 path")
p2_line, = ax.plot([], [], lw=2, label="p2 path")

# Current positions as scatter points
p1_point = ax.scatter([p1_0[0]], [p1_0[1]], s=70, label="p1")
p2_point = ax.scatter([p2_0[0]], [p2_0[1]], s=70, label="p2")

# Arrow showing coupling p2 -> p1 (initialized to valid coords)
arrow = FancyArrowPatch(tuple(p2_0), tuple(p1_0),
                        arrowstyle="->", mutation_scale=15, linewidth=1.5)
ax.add_patch(arrow)

# Text overlays
time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
err_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, va="top")

ax.legend(loc="best")

def init():
    p1_line.set_data([], [])
    p2_line.set_data([], [])
    p1_point.set_offsets(p1_0)
    p2_point.set_offsets(p2_0)
    arrow.set_positions(tuple(p2_0), tuple(p1_0))
    time_text.set_text("t = 0.00 s")
    e0 = np.linalg.norm(p2_0 - g)
    err_text.set_text(f"||p2 - g|| = {e0:.3f}")
    return p1_line, p2_line, p1_point, p2_point, arrow, time_text, err_text

def update(k):
    i = frame_ids[k]

    # Trails
    p1_line.set_data(p1_traj[:i+1, 0], p1_traj[:i+1, 1])
    p2_line.set_data(p2_traj[:i+1, 0], p2_traj[:i+1, 1])

    # Current points
    p1_point.set_offsets(p1_traj[i])
    p2_point.set_offsets(p2_traj[i])

    # Coupling arrow p2 -> p1
    arrow.set_positions(tuple(p2_traj[i]), tuple(p1_traj[i]))

    # Text
    time_text.set_text(f"t = {time[i]:.2f} s")
    err = np.linalg.norm(p2_traj[i] - g)
    err_text.set_text(f"||p2 - g|| = {err:.3f}")

    return p1_line, p2_line, p1_point, p2_point, arrow, time_text, err_text

ani = FuncAnimation(
    fig,
    update,
    frames=len(frame_ids),
    init_func=init,
    blit=False,          # robust across backends
    interval=30
)

# Save GIF (requires pillow: pip install pillow)
out_gif = "two_point_backstepping.gif"
ani.save(out_gif, writer="pillow", fps=30)

plt.close(fig)
print(f"Saved GIF to: {out_gif}")
