"""
Generate a catchy multi-panel figure + transient video of the GPMF-style simulation.
"""
from __future__ import print_function, division
from fenics import *
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import os

# ============================================================
# Parameters
# ============================================================
H = 10.0; WIDTH = 0.1
k = 5e-15; mu_f = 1e-3
K = 1e8; G = 6e7
Gc = Constant(G); Kc = Constant(K)
alpha = 1.0; phi_0 = 0.3
dt_val = 5000.0
nsteps = 57
sigma_load = 1.0e4

w_inf = Constant(0.6)
w_k = [Constant(0.4)]
tau_k = [Constant(1e4)]

C2 = Constant(phi_0 * 1e-9)
C1 = Constant(alpha)

# ============================================================
# Mesh & Function Spaces
# ============================================================
mesh = BoxMesh(Point(0,0,0), Point(H,WIDTH,WIDTH), 50, 1, 1)

ele_p = FiniteElement("DG", mesh.ufl_cell(), 0)
ele_q = FiniteElement("BDM", mesh.ufl_cell(), 1)
ele_us = VectorElement("CG", mesh.ufl_cell(), 1)
W_elem = MixedElement([ele_p, ele_q, ele_us])
W = FunctionSpace(mesh, W_elem)
S = TensorFunctionSpace(mesh, "DG", 0)

# ============================================================
# Boundary Conditions
# ============================================================
class Top(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary
class Base(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], H) and on_boundary
class Sides(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1],0.0) or near(x[1],WIDTH) or near(x[2],0.0) or near(x[2],WIDTH)) and on_boundary

mf = MeshFunction("size_t", mesh, 2)
mf.set_all(0); Top().mark(mf,1); Base().mark(mf,2); Sides().mark(mf,3)
ds = Measure("ds")(subdomain_data=mf)

bcs = [
    DirichletBC(W.sub(0), Constant(0.0), mf, 1),           # drained top
    DirichletBC(W.sub(2).sub(0), Constant(0.0), mf, 2),    # fix base
    DirichletBC(W.sub(2).sub(1), Constant(0.0), mf, 2),
    DirichletBC(W.sub(2).sub(2), Constant(0.0), mf, 2),
    DirichletBC(W.sub(2).sub(1), Constant(0.0), mf, 3),    # u_y=0 on sides
    DirichletBC(W.sub(2).sub(2), Constant(0.0), mf, 3),    # u_z=0 on sides
]

# ============================================================
# Initialize (psi must be created before forms)
# ============================================================
psi = Function(S)
psi.vector()[:] = 0.0
X = Function(W)

X_n = interpolate(Expression(("0","0","0","0","0","0","0"), degree=2), W)

# ============================================================
# Variational Forms
# ============================================================
U = TrialFunction(W); V = TestFunction(W)
p, q, us = split(U); Pt, Qt, Ust = split(V)
p_n, q_n, us_n = split(X_n)

def epsilon(w): return 0.5*(grad(w)+grad(w).T)
def epsilon_v(w): return div(w)

# Mass conservation
CoMass_l = C1*epsilon_v(us)*Pt*dx + C2*p*Pt*dx + Constant(dt_val)*div(q)*Pt*dx
CoMass_r = C1*epsilon_v(us_n)*Pt*dx + C2*p_n*Pt*dx

# Darcy
DL_l = Constant(mu_f/k)*inner(q, Qt)*dx - p*div(Qt)*dx

# Momentum
w_total = w_inf + w_k[0]
sigma_el = (-2*w_total*Gc*epsilon(us) - w_total*Constant(K-2/3.*G)*epsilon_v(us)*Identity(3))
sigma_rel = -w_k[0]*(2*Gc*psi + Constant(K-2/3.*G)*tr(psi)*Identity(3))

traction = inner(Constant((-sigma_load, 0., 0.)), Ust)*ds(1)

CoMom_el = inner(sigma_el + Constant(alpha)*p*Identity(3), grad(Ust))*dx
CoMom_rel = inner(sigma_rel, grad(Ust))*dx

A_form = CoMass_l + DL_l + CoMom_el
L_form = CoMass_r + CoMom_rel + traction

Nx = 100
z_vals = np.linspace(0, H, Nx)

c_v = k / mu_f * (K + 4/3.*G)
tau_c = 1e4 * (K + 4/3.*G) / (w_inf * (K + 4/3.*G))
E_oe = K + 4/3.*G
u_drained = sigma_load * H / E_oe
u_ve_limit = sigma_load * H / (float(w_inf) * E_oe)

# Storage
p_profiles = {}
u_hist = []
t_hist = []
p_mid_hist = []
saved_tau = [0.05, 0.1, 0.2, 0.5, 1.0]

outdir = "papers/poro_viscoelastic_fenics/figures"
os.makedirs(os.path.join(outdir, "frames"), exist_ok=True)

# ============================================================
# Time Loop
# ============================================================
print("Running simulation...")
for n in range(nsteps):
    t = (n+1) * dt_val
    tau_val = c_v * t / H**2
    
    A_mat, b_vec = assemble_system(A_form, L_form, bcs)
    solve(A_mat, X.vector(), b_vec, "mumps")
    
    p_out, q_out, us_out = X.split(True)
    u_top = us_out(Point(0., WIDTH/2, WIDTH/2))[0]
    p_mid = p_out(Point(H/2, WIDTH/2, WIDTH/2))
    
    u_hist.append(u_top)
    t_hist.append(t)
    p_mid_hist.append(p_mid)
    
    # Extract pressure profile
    p_vals = np.array([p_out(Point(z, WIDTH/2, WIDTH/2)) for z in z_vals])
    
    # Check if we're near a saved tau
    for tau_target in saved_tau:
        if abs(tau_val - tau_target) < 1e-3 or (tau_target not in p_profiles and tau_val > tau_target):
            if tau_target not in p_profiles:
                p_profiles[tau_target] = (z_vals.copy(), p_vals.copy(), t)
                print("  tau={:.2f} profile saved at t={:.1f}s".format(tau_target, t))
    
    # Update viscous strain
    eps_new = epsilon(us_out)
    factor = float(tau_k[0]) / (float(tau_k[0]) + dt_val)
    psi_new = project(factor*psi + (1-factor)*eps_new, S)
    psi.assign(psi_new)
    X_n.assign(X)
    
    # Generate frame every 5 steps
    if n % 5 == 0 or n == nsteps - 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        
        # Panel 1: Pressure profile
        ax1 = axes[0]
        cmap = plt.cm.viridis
        tau_list = sorted(p_profiles.keys())
        for i, tau_val_p in enumerate(tau_list):
            zz, pp, _ = p_profiles[tau_val_p]
            pp_norm = pp / sigma_load if sigma_load > 0 else pp
            color = cmap(i / max(len(tau_list)-1, 1))
            ax1.plot(pp_norm, zz/H, color=color, lw=2, label=r"$\tau={:.2f}$".format(tau_val_p))
        ax1.set_xlabel(r"$p/p_0$", fontsize=12)
        ax1.set_ylabel(r"$z/H$", fontsize=12)
        ax1.set_title("Pore Pressure Profiles", fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # Panel 2: Settlement curve
        ax2 = axes[1]
        ax2.plot(t_hist, np.array(u_hist)*1000, 'b-', lw=2, label="Numerical")
        ax2.axhline(y=u_drained*1000, color='gray', ls='--', lw=1.5,
                    label=r"Drained elastic: $u_d$")
        ax2.axhline(y=u_ve_limit*1000, color='r', ls='--', lw=1.5,
                    label=r"VE limit: $u_\infty$")
        ax2.fill_between(t_hist, u_drained*1000, np.array(u_hist)*1000,
                         alpha=0.15, color='r', label="VE enhancement")
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("Surface displacement (mm)", fontsize=12)
        ax2.set_title("Consolidation Settlement", fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Pore pressure at mid-height
        ax3 = axes[2]
        ax3.semilogy(t_hist, np.maximum(np.abs(p_mid_hist), 1e-8), 'g-', lw=2)
        ax3.set_xlabel("Time (s)", fontsize=12)
        ax3.set_ylabel(r"$|p|$ at mid-height (Pa)", fontsize=12)
        ax3.set_title("Pore Pressure Dissipation", fontsize=13)
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle("Poro-Viscoelastic Consolidation (t={:.1f}s, {:.2f}% of VE limit)".format(
            t, 100*u_top/u_ve_limit), fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "frames", "frame_{:04d}.png".format(n)),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Frame {:04d} / {:04d} (t={:.1f}s, u={:.4f}mm)".format(
            n+1, nsteps, t, u_top*1000))

# ============================================================
# Composite "Catchy" Figure
# ============================================================
print("\nGenerating composite figure...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# --- Top-left: Pressure profiles ---
ax1 = fig.add_subplot(gs[0, 0])
cmap = plt.cm.viridis
tau_list = sorted(p_profiles.keys())
for i, tau_val_p in enumerate(tau_list):
    zz, pp, _ = p_profiles[tau_val_p]
    pp_norm = pp / sigma_load
    color = cmap(i / max(len(tau_list)-1, 1))
    ax1.plot(pp_norm, zz/H, color=color, lw=2.5,
             label=r"$\tau={:.2f}$".format(tau_val_p))
ax1.set_xlabel(r"$p/p_0$", fontsize=14)
ax1.set_ylabel(r"$z/H$", fontsize=14)
ax1.set_title("(a) Pore Pressure Evolution", fontsize=15)
ax1.legend(fontsize=10, loc="lower right")
ax1.grid(True, alpha=0.3)
ax1.invert_yaxis()
ax1.tick_params(labelsize=11)

# --- Top-right: Settlement ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t_hist, np.array(u_hist)*1000, 'b-', lw=2.5, label="Numerical")
ax2.axhline(y=u_drained*1000, color='gray', ls='--', lw=2,
            label=r"Drained elastic ($u_d$)")
ax2.axhline(y=u_ve_limit*1000, color='r', ls='--', lw=2,
            label=r"VE creep limit ($u_\infty$)")
ax2.fill_between(t_hist, u_drained*1000, np.array(u_hist)*1000,
                 alpha=0.2, color='crimson',
                 label=r"${:.1f}\%$ enhancement".format(100*(u_ve_limit/u_drained-1)))
ax2.set_xlabel("Time (s)", fontsize=14)
ax2.set_ylabel("Surface displacement (mm)", fontsize=14)
ax2.set_title("(b) Settlement Curve", fontsize=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=11)

# --- Top-right (2): Consolidation curve U vs tau ---
ax2b = fig.add_subplot(gs[0, 2])
u_arr = np.array(u_hist)
U_vals = np.zeros_like(u_arr)
# U = (u(t) - u(0)) / (u(inf) - u(0))
# But u(0) is the undrained displacement (very small), use drained as reference
u0 = u_hist[0] if len(u_hist) > 0 else 0
U_vals = (u_arr - u0) / (u_ve_limit - u0)
tau_plot = c_v * np.array(t_hist) / H**2
ax2b.plot(tau_plot, U_vals*100, 'b-', lw=2.5, label="Numerical")
# Terzaghi analytical U
tau_dense = np.logspace(-3, 2, 200)
U_ana = np.zeros_like(tau_dense)
for m in range(20):
    M = np.pi/2 * (2*m+1)
    U_ana += 2/M**2 * (1 - np.exp(-M**2 * tau_dense))
ax2b.plot(tau_dense, U_ana*100, 'k--', lw=1.5, alpha=0.7, label="Terzaghi (elastic)")
# VE limit
ax2b.axhline(y=100, color='r', ls=':', lw=2, label="VE limit (U=100%)")
ax2b.set_xscale("log")
ax2b.set_xlabel(r"$\tau = c_v t / H^2$", fontsize=14)
ax2b.set_ylabel(r"Consolidation $U$ (%)", fontsize=14)
ax2b.set_title("(c) Degree of Consolidation", fontsize=15)
ax2b.legend(fontsize=10)
ax2b.grid(True, alpha=0.3)
ax2b.tick_params(labelsize=11)

# --- Bottom row: 3D column visualization ---
# Create a pseudocolor plot of the column at the final state
ax3 = fig.add_subplot(gs[1, :])
z_grid = np.linspace(0, H, 100)
u_x_vals = np.array([us_out(Point(z, WIDTH/2, WIDTH/2))[0] for z in z_grid])
u_y_vals = np.array([us_out(Point(z, WIDTH/2, WIDTH/2))[1] for z in z_grid])
u_z_vals = np.array([us_out(Point(z, WIDTH/2, WIDTH/2))[2] for z in z_grid])
u_mag = np.sqrt(u_x_vals**2 + u_y_vals**2 + u_z_vals**2) * 1000  # in mm
p_final = np.array([p_out(Point(z, WIDTH/2, WIDTH/2)) for z in z_grid])

ax3b = ax3.twinx()
# Plot displacement profile
line1 = ax3.plot(u_x_vals*1000, z_grid, 'b-', lw=2.5, label=r"$u_x$ (mm)")
line2 = ax3b.plot(p_final, z_grid, 'r--', lw=2.5, label=r"$p$ (Pa)")
# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, fontsize=11, loc="lower left")

ax3.set_xlabel(r"$u_x$ (mm) / $p$ (Pa)", fontsize=14)
ax3.set_ylabel(r"$z$ (m)", fontsize=14)
ax3.set_title("(d) Final State: Displacement and Pore Pressure Profiles ($t={:.0f}$s)".format(
    t_hist[-1]), fontsize=15)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=11)
ax3b.tick_params(labelsize=11)
ax3.invert_yaxis()
ax3.set_ylim(H, 0)
ax3b.set_ylim(H, 0)

# Add annotations
E_oe_v = K + 4/3.*G
E_ve = float(w_inf) * E_oe_v
ax3.annotate(r"$E_{oe}=%.2e$ Pa" % E_oe_v, xy=(0.05, 0.85), xycoords="axes fraction",
             fontsize=11, bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
ax3.annotate(r"$E_{VE}=%.2e$ Pa" % E_ve, xy=(0.05, 0.75), xycoords="axes fraction",
             fontsize=11, bbox=dict(boxstyle="round", fc="lightcoral", alpha=0.8))
ax3.annotate(r"Enhancement: $E_{oe}/E_{VE}=%.3f$" % (E_oe_v/E_ve),
             xy=(0.05, 0.65), xycoords="axes fraction",
             fontsize=11, bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.8))

plt.savefig(os.path.join(outdir, "fig-catchy.png"), dpi=200, bbox_inches="tight")
print("Saved: fig-catchy.png")

# ============================================================
# Generate Video
# ============================================================
print("\nGenerating video from frames...")
os.system("cd {} && ffmpeg -y -framerate 5 -pattern_type glob -i 'frames/*.png' "
          "-c:v libx264 -pix_fmt yuv420p -crf 20 ../simulation.mp4 2>/dev/null".format(outdir))
print("Saved: simulation.mp4")

print("\n=== Done ===")
print("Final displacement: {:.4f} mm".format(u_hist[-1]*1000))
print("Drained elastic:    {:.4f} mm".format(u_drained*1000))
print("VE creep limit:     {:.4f} mm".format(u_ve_limit*1000))
