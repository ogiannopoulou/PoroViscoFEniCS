"""
PoroViscoFEniCS

A Generalized Poro-Viscoelastic Model using FEniCS

Extends the Generalized Poroelastic Model (GPMF, Haagenson et al., 2020,
Computers & Geosciences 135, 104399) by adding viscoelastic creep behavior
via the Generalized Maxwell (Wiechert) model.

Primary unknowns:  p  (pore pressure perturbation, DG(0))
                    q  (fluid flux, BDM(1))
                    us (solid displacement, CG(1) vector)

Internal variables: epsilon_v_k (viscous strain, DG(0) tensor) per branch

Governing equations:
  - Conservation of momentum:     div(sigma) + f = 0
  - Conservation of fluid mass:   dzeta/dt + div(q) = 0
  - Darcy's law:                  q = -(k/mu) grad(p)
  - Viscoelastic creep (branch k):
        d(epsilon_v_k)/dt = (C_k : epsilon - C_k : epsilon_v_k) / eta_k

Constitutive:
  - Total stress:       sigma = sigma_eff - alpha p I
  - Effective stress:   sigma_eff = C_inf : epsilon
                                  + sum_k C_k : (epsilon - epsilon_v_k)
  - Fluid content:      zeta = alpha div(us)
                                  + (alpha - phi) beta_s p + phi beta_f p

References:
  Haagenson, R., Rajaram, H., Allen, J. (2020).
  "A generalized poroelastic model using FEniCS with insights into
   the Noordbergum effect." Computers & Geosciences, 135, 104399.

  Nespoli, M., et al. (2023). "Poro-viscoelastic modelling of fluid
  injection in layered media." Geophysical Journal International.
"""

from __future__ import print_function, division
from fenics import *
import numpy as np
import ufl
from time import gmtime, strftime

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

if MPI.rank(MPI.comm_world) == 0:
    print("PoroViscoFEniCS — Generalized Poro-Viscoelastic Model")
    print("Start date and time:", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("")

# ==========================================================================
# MODEL TYPE
# ==========================================================================

Linear = 1      # 1 = linear, 0 = not linear
Nonlinear = 0   # 1 = nonlinear (Picard), 0 = not nonlinear
                # (Nonlinear places holder for future strain-dependent props)

if Linear + Nonlinear != 1:
    if MPI.rank(MPI.comm_world) == 0:
        print("You must choose a model type (Linear or Nonlinear).")
    quit()

if Linear == 1:
    if MPI.rank(MPI.comm_world) == 0:
        print("Running the Linear Poro-Viscoelastic Model.")
    weight = Constant(0.0)
    linear_flag = 1

if Nonlinear == 1:
    if MPI.rank(MPI.comm_world) == 0:
        print("Running the Nonlinear Poro-Viscoelastic Model.")
    weight = Constant(1.0)
    linear_flag = 0

# ==========================================================================
# DOMAIN AND SUBDOMAINS
# ==========================================================================

x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 5.0
z0 = 0.0
z1 = 5.0
nx = 30
ny = 15
nz = 15

# ==========================================================================
# MESHING
# ==========================================================================

if MPI.rank(MPI.comm_world) == 0:
    print("Building mesh...")
mesh = BoxMesh(Point(x0, y0, z0), Point(x1, y1, z1), nx, ny, nz)

# ==========================================================================
# FUNCTION SPACES
# ==========================================================================

ele_p  = FiniteElement("DG",  mesh.ufl_cell(), 0)   # pressure
ele_q  = FiniteElement("BDM", mesh.ufl_cell(), 1)   # fluid flux
ele_us = VectorElement("CG",  mesh.ufl_cell(), 1)   # solid displacement

W = MixedElement([ele_p, ele_q, ele_us])
W = FunctionSpace(mesh, W)

Z = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "DG", 0)
S = TensorFunctionSpace(mesh, "DG", 0)    # for viscous strain (symmetric)
P = FunctionSpace(mesh, "DG", 0)

# ==========================================================================
# PHYSICAL PARAMETERS
# ==========================================================================

g = 9.81                                 # Gravity            (m/s^2)

k = 1.0e-12                              # Permeability       (m^2)
mu = 1.0e-3                              # Viscosity          (Pa*s)
phi_0 = 0.1                              # Initial porosity   (-)
phi_min = 0.01                           # Minimum porosity
p_0 = 0.0                                # Reference pressure  (Pa)
rho_0 = 1000.0                           # Reference density   (kg/m^3)

beta_m = 1.0e-9                          # Matrix compressibility (Pa^-1)
beta_f = 4.5e-10                         # Fluid compressibility  (Pa^-1)
beta_s = 1.0e-10                         # Solid compressibility  (Pa^-1)
nu = 0.25                                # Poisson's ratio    (-)

K = beta_m**(-1)                         # Bulk modulus       (Pa)
G = 1.5 * K * (1.0 - 2.0*nu) / (1.0+nu) # Shear modulus      (Pa)
alpha = 1.0 - beta_s / beta_m            # Biot coefficient   (-)

if MPI.rank(MPI.comm_world) == 0:
    print("Elastic parameters: K = {:.3e} Pa, G = {:.3e} Pa, alpha = {:.3f}"
          .format(float(K), float(G), float(alpha)))

# ==========================================================================
# VISCOELASTIC PARAMETERS (Generalized Maxwell / Wiechert model)
# ==========================================================================

# Weights: w_inf is the equilibrium (long-term) fraction;
# w_k for each Maxwell branch are the non-equilibrium fractions.
# Sum(w_inf) + Sum(w_k) = 1
#
# Relaxation times tau_k = eta_k / G_k  for each branch.
#
# The instantaneous shear modulus is G (the elastic value from above).
# Each branch k has shear modulus G_k = w_k * G and viscosity eta_k = tau_k * G_k.

n_ve_branches = 2                         # number of Maxwell branches

w_inf = Constant(0.5)                     # equilibrium stiffness fraction
w = [Constant(0.3), Constant(0.2)]       # branch stiffness fractions (sum + w_inf = 1)

tau = [Constant(1.0e3), Constant(1.0e5)] # relaxation times (s)

# Derived branch viscosities
eta = [tau[i] * w[i] * G for i in range(n_ve_branches)]

if MPI.rank(MPI.comm_world) == 0:
    print("Viscoelastic: {} Maxwell branches, w_inf = {:.3f}"
          .format(n_ve_branches, float(w_inf)))
    for i in range(n_ve_branches):
        print("  Branch {}: w = {:.3f}, tau = {:.2e} s, eta = {:.2e} Pa*s"
              .format(i+1, float(w[i]), float(tau[i]), float(eta[i])))

# ==========================================================================
# HYDROSTATIC CONDITION
# ==========================================================================

p_h = project(
    Expression("rho_0 * g * (z1 - x[2])",
               degree=1, rho_0=rho_0, g=g, z1=z1),
    Z
)

# ==========================================================================
# CONSTITUTIVE FUNCTIONS
# ==========================================================================

def epsilon(us):
    """Infinitesimal strain tensor: sym(grad(us))."""
    return 0.5 * (grad(us) + grad(us).T)

def epsilon_v(us):
    """Volumetric strain: div(us)."""
    return div(us)

def sigma_e_elastic(us, K, G):
    """Elastic effective stress (negative sign: compression-positive convention)."""
    eps = epsilon(us)
    eps_vl = epsilon_v(us)
    return -2.0 * G * eps - (K - 2.0/3.0 * G) * eps_vl * Identity(len(us))

def sigma_e_visco(us, epsilon_v_branches, K, G, w_inf, w_branches):
    """
    Viscoelastic effective stress using Generalized Maxwell model.

    sigma_eff = C_inf : epsilon + sum_k C_k : (epsilon - epsilon_v_k)

    where:
      C_inf = w_inf * C
      C_k   = w_k * C
      C     = 2G*dev(I four I) + K*(I x I)   (elasticity tensor)
    """
    ndim = len(us)
    I = Identity(ndim)
    eps = epsilon(us)
    eps_vl = epsilon_v(us)

    # Equilibrium (long-term) contribution
    sigma_val = (-2.0 * w_inf * G * eps
                 - w_inf * (K - 2.0/3.0 * G) * eps_vl * I)

    # Non-equilibrium (viscous) branches
    for k in range(n_ve_branches):
        w_k = w_branches[k]
        eps_k = epsilon_v_branches[k]
        eps_k_vl = tr(eps_k)

        sigma_val += (-2.0 * w_k * G * (eps - eps_k)
                      - w_k * (K - 2.0/3.0 * G) * (eps_vl - eps_k_vl) * I)

    return sigma_val

def sigma_total(us, epsilon_v_branches, p, alpha, K, G, w_inf, w_branches):
    """Total stress: sigma = sigma_eff - alpha * p * I."""
    ndim = len(us)
    I = Identity(ndim)
    return sigma_e_visco(us, epsilon_v_branches, K, G, w_inf, w_branches) \
           + alpha * p * I

# Linear version: constant fluid density and porosity
def rho_f(p, p_0, p_h, rho_0, beta_f):
    return Constant(rho_0)

def phi_func(alpha, us, p, beta_s, phi_0, phi_min, t):
    return Constant(phi_0)

# ==========================================================================
# TIME PARAMETERS
# ==========================================================================

tend = 1.0e5           # total simulation time (s)
nsteps = 20             # number of time steps
dt = tend / nsteps      # time step size (s)

if MPI.rank(MPI.comm_world) == 0:
    print("Time: tend = {:.2e} s, dt = {:.2e} s, steps = {}"
          .format(tend, dt, nsteps))

# ==========================================================================
# INITIAL CONDITIONS
# ==========================================================================

X_i = Expression(
    ("0.0",                          # p
     "0.0", "0.0", "0.0",           # (q1, q2, q3)
     "0.0", "0.0", "0.0"),          # (us1, us2, us3)
    degree=2
)
X_n = interpolate(X_i, W)            # solution at previous time step

phi_n = interpolate(Constant(phi_0), Z)

# Viscous strain for each Maxwell branch (initialized to zero)
epsilon_v_branches_n = []
for k in range(n_ve_branches):
    eps_k = Function(S, name="epsilon_v_{}_n".format(k))
    eps_k.vector()[:] = 0.0
    epsilon_v_branches_n.append(eps_k)

epsilon_v_branches = []
for k in range(n_ve_branches):
    eps_k = Function(S, name="epsilon_v_{}".format(k))
    eps_k.vector()[:] = 0.0
    epsilon_v_branches.append(eps_k)

# ==========================================================================
# BOUNDARY CONDITIONS
# ==========================================================================

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], x0)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], x1)

class BackBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], y0)

class FrontBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], y1)

class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], z0)

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], z1)

boundary_facet_function = MeshFunction("size_t", mesh, 2)
boundary_facet_function.set_all(0)
LeftBoundary().mark(boundary_facet_function, 1)
RightBoundary().mark(boundary_facet_function, 2)
BackBoundary().mark(boundary_facet_function, 3)
FrontBoundary().mark(boundary_facet_function, 4)
BottomBoundary().mark(boundary_facet_function, 5)
TopBoundary().mark(boundary_facet_function, 6)


def GetBoundaryConditions(t):
    """
    3D oedometric column test:
      - Top (z=z1): drained p=0
      - Base (z=z0): u=(0,0,0), impermeable
      - x-sides (x=x0, x1): u_x=0 (roller)
      - y-sides (y=y0, y1): u_y=0 (roller)
    """
    bcs = []
    bcs.append(DirichletBC(W.sub(0), Constant(0.0), boundary_facet_function, 6))
    bcs.append(DirichletBC(W.sub(2), Constant((0.0, 0.0, 0.0)), boundary_facet_function, 5))
    bcs.append(DirichletBC(W.sub(2).sub(0), Constant(0.0), boundary_facet_function, 1))
    bcs.append(DirichletBC(W.sub(2).sub(0), Constant(0.0), boundary_facet_function, 2))
    bcs.append(DirichletBC(W.sub(2).sub(1), Constant(0.0), boundary_facet_function, 3))
    bcs.append(DirichletBC(W.sub(2).sub(1), Constant(0.0), boundary_facet_function, 4))
    return bcs


# ==========================================================================
# SOLVER SETUP
# ==========================================================================

U = TrialFunction(W)
V = TestFunction(W)

n = FacetNormal(mesh)
norm = as_vector([n[0], n[1], n[2]])

ff = Constant(0.0)   # fluid source/sink

X = Function(W)

density_save = Function(Z)
porosity_save = Function(Z)

ds = Measure("ds")(subdomain_data=boundary_facet_function)
ones_func = project(Constant(1.0), Z, solver_type="gmres")


def sigma_elastic(us, K, G):
    """Elastic effective stress (trial function only)."""
    ndim = len(us)
    I = Identity(ndim)
    eps = epsilon(us)
    eps_vl = epsilon_v(us)
    return -2.0 * G * eps - (K - 2.0/3.0 * G) * eps_vl * I

def sigma_relaxation(K, G):
    """
    Viscoelastic relaxation stress from frozen viscous strains.
    Returns sum_k w_k * C : eps_v_k  (coefficient-only, no arguments).
    """
    ndim = 3
    I = Identity(ndim)
    if n_ve_branches == 0:
        return Constant(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    # Minus sign: same convention as sigma_elast (compression-positive)
    sigma_val = - w[0] * (2.0 * G * epsilon_v_branches_n[0]
                          + (K - 2.0/3.0 * G) * tr(epsilon_v_branches_n[0]) * I)
    for k in range(1, n_ve_branches):
        w_k = w[k]
        eps_k = epsilon_v_branches_n[k]
        sigma_val += - w_k * (2.0 * G * eps_k
                              + (K - 2.0/3.0 * G) * tr(eps_k) * I)
    return sigma_val

def WeakForm(U, V, X_n, t):
    """
    Assemble the weak form for the poro-viscoelastic system.

    LHS: elastic + pressure terms (trial functions us, p)
    RHS: viscoelastic relaxation (frozen viscous strains) + traction

    Variables:
      U   = (p, q, us)        -- trial functions
      V   = (Pt, Qt, Ust)     -- test functions
      X_n = (p_n, q_n, us_n)  -- previous time step solution
    """
    p, q, us = split(U)
    Pt, Qt, Ust = split(V)
    p_n, q_n, us_n = split(X_n)

    # --- Conservation of Mass --------------------------------------------
    CoMass_l_1 = (rho_f(p_n, p_0, p_h, rho_0, beta_f)
                  * Constant(alpha) * epsilon_v(us) * Pt * dx)
    CoMass_l_2 = (rho_f(p_n, p_0, p_h, rho_0, beta_f)
                  * ((Constant(alpha) - phi_func(alpha, us_n, p_n, beta_s,
                                                  phi_0, phi_min, t))
                     * Constant(beta_s)
                     + phi_func(alpha, us_n, p_n, beta_s, phi_0, phi_min, t)
                     * Constant(beta_f))
                  * p * Pt * dx)
    CoMass_l_3 = dt * rho_f(p_n, p_0, p_h, rho_0, beta_f) * div(q) * Pt * dx
    CoMass_l_4 = (dt * Constant(weight)
                  * inner(q, grad(rho_f(p_n, p_0, p_h, rho_0, beta_f)
                                  * ones_func))
                  * Pt * dx)
    CoMass_l = CoMass_l_1 + CoMass_l_2 + CoMass_l_3 + CoMass_l_4

    CoMass_r_1 = dt * ff * Pt * dx
    CoMass_r_2 = (rho_f(p_n, p_0, p_h, rho_0, beta_f)
                  * Constant(alpha) * epsilon_v(us_n) * Pt * dx)
    CoMass_r_3 = (rho_f(p_n, p_0, p_h, rho_0, beta_f)
                  * ((Constant(alpha) - phi_func(alpha, us_n, p_n, beta_s,
                                                  phi_0, phi_min, t))
                     * Constant(beta_s)
                     + phi_func(alpha, us_n, p_n, beta_s, phi_0, phi_min, t)
                     * Constant(beta_f))
                  * p_n * Pt * dx)
    CoMass_r = CoMass_r_1 + CoMass_r_2 + CoMass_r_3

    # --- Darcy's Law -----------------------------------------------------
    DL_l = mu / k * inner(q, Qt) * dx - p * div(Qt) * dx

    # --- Conservation of Momentum ----------------------------------------
    # LHS: elastic + pressure (trial -> bilinear form)
    CoMom_elastic = inner(sigma_elastic(us, K, G), grad(Ust)) * dx
    CoMom_pressure = inner(Constant(alpha) * p * Identity(3), grad(Ust)) * dx

    # RHS: viscoelastic relaxation (coefficients -> linear form)
    CoMom_relax = inner(sigma_relaxation(K, G), grad(Ust)) * dx

    # Traction: compressive load on top boundary (facet 6)
    traction = inner(Constant((0.0, 0.0, -1.0e4)), Ust) * ds(6)

    A = CoMass_l + DL_l + CoMom_elastic + CoMom_pressure
    B = CoMass_r + CoMom_relax + traction

    return A, B


def LinearSolver(U, V, X_n, t, bcs):
    a, L = WeakForm(U, V, X_n, t)
    A_mat, b_vec = assemble_system(a, L, bcs)
    solve(A_mat, X.vector(), b_vec, "mumps")
    return X


# ==========================================================================
# VISCOELASTIC STRAIN UPDATE
# ==========================================================================

def update_viscous_strain(epsilon_v_old, us_new, dt_branch, tau_branch):
    """
    Semi-implicit (backward Euler) update of viscous strain for one branch.

    d(epsilon_v)/dt + (1/tau) * epsilon_v = (1/tau) * epsilon

    Backward Euler:
      epsilon_v_new = (tau/(tau+dt)) * epsilon_v_old
                    + (dt/(tau+dt)) * epsilon(us_new)
    """
    eps_new = epsilon(us_new)
    factor_old = tau_branch / (tau_branch + dt_branch)
    factor_new = dt_branch / (tau_branch + dt_branch)

    eps_v_new = factor_old * epsilon_v_old + factor_new * eps_new
    return eps_v_new


# ==========================================================================
# OUTPUT FILES
# ==========================================================================

import os
output_dir = "PoroViscoResults"
if MPI.rank(MPI.comm_world) == 0:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

pressure_file = XDMFFile(os.path.join(output_dir, "pressure.xdmf"))
flux_file = XDMFFile(os.path.join(output_dir, "flux.xdmf"))
disp_file = XDMFFile(os.path.join(output_dir, "disp.xdmf"))
density_file = XDMFFile(os.path.join(output_dir, "density.xdmf"))
porosity_file = XDMFFile(os.path.join(output_dir, "porosity.xdmf"))
stress_file = XDMFFile(os.path.join(output_dir, "stress.xdmf"))

# Viscous strain output for each branch
viscous_strain_files = []
for k in range(n_ve_branches):
    f = XDMFFile(os.path.join(output_dir, "epsilon_v_{}.xdmf".format(k)))
    viscous_strain_files.append(f)


# ==========================================================================
# TIME LOOP
# ==========================================================================

t = 0.0

if MPI.rank(MPI.comm_world) == 0:
    print("")
    print("Starting Time Loop...")
    print("")

for n in range(nsteps):
    MPI.barrier(MPI.comm_world)
    t += dt

    if MPI.rank(MPI.comm_world) == 0:
        print("###############")
        print("")
        print("NEW TIME STEP")
        print("Time = {:.4e} s".format(t))
        print("Progress = {:.4f}%".format(t / tend * 100.0))
        print("")

    # Convergence criteria (for future nonlinear extension)
    reltol = 1e-4
    rel_error_max_global = 9999.0
    max_iter = 100
    omega = 1.0
    iter_count = 0

    bcs = GetBoundaryConditions(t)

    # ----- Solve poro-viscoelastic system --------------------------------
    # For the linear model, a single solve suffices

    if MPI.rank(MPI.comm_world) == 0:
        print("Solving poro-viscoelastic system...")

    # Copy current viscous strain to "previous" state for the weak form
    for k in range(n_ve_branches):
        epsilon_v_branches_n[k].vector()[:] = epsilon_v_branches[k].vector()

    X = LinearSolver(U, V, X_n, t, bcs)

    if MPI.rank(MPI.comm_world) == 0:
        print("System solved.")

    # Split solution
    p, q, us = X.split(True)

    # ----- Update viscous strains (semi-implicit) ------------------------
    if MPI.rank(MPI.comm_world) == 0:
        print("Updating viscous strains...")

    for k in range(n_ve_branches):
        # Compute new viscous strain for branch k
        eps_v_new_expr = project(
            update_viscous_strain(epsilon_v_branches_n[k], us,
                                  Constant(dt), tau[k]),
            S
        )
        epsilon_v_branches[k].assign(eps_v_new_expr)

    if MPI.rank(MPI.comm_world) == 0:
        print("Viscous strains updated.")

    # ----- Compute total stress for output -------------------------------
    w_branches_list = [w[k] for k in range(n_ve_branches)]
    eps_v_branches_list = [epsilon_v_branches[k] for k in range(n_ve_branches)]

    sigma_expr = project(
        sigma_total(us, eps_v_branches_list, p, alpha, K, G,
                    w_inf, w_branches_list),
        S
    )

    # ----- Save results --------------------------------------------------
    if MPI.rank(MPI.comm_world) == 0:
        print("Saving results...")
        print("")

    pressure_file.write(p, t)
    flux_file.write(q, t)
    disp_file.write(us, t)
    stress_file.write(sigma_expr, t)

    density_save.assign(project(rho_f(p, p_0, p_h, rho_0, beta_f), P))
    porosity_save.assign(
        project(phi_func(alpha, us, p, beta_s, phi_0, phi_min, t), P)
    )
    density_file.write(density_save, t)
    porosity_file.write(porosity_save, t)

    for k in range(n_ve_branches):
        viscous_strain_files[k].write(epsilon_v_branches[k], t)

    # ----- Update solution for next time step ----------------------------
    X_n.assign(X)

    if MPI.rank(MPI.comm_world) == 0:
        print("Solution updated for next time step.")
        print("###############")
        print("")

# ==========================================================================
# FINISH
# ==========================================================================

if MPI.rank(MPI.comm_world) == 0:
    print("")
    print("PoroViscoFEniCS simulation complete.")
    print("End date and time:", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Results written to: {}".format(output_dir))
