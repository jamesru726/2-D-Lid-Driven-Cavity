import numpy as np
import math
import ctypes
import matplotlib.pyplot as plt
import sys
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=3)

# Builds A coefficient matrices
def Build_A(A_N, A_W, A_C, A_E, A_S, u_face, v_face, mdot_n, mdot_w, mdot_e, mdot_s, AN_c, AW_c, AC_c, AE_c, AS_c):

    A_N[1:-1, 1:-1] = -AN_D
    A_W[1:-1, 1:-1] = -AW_D
    A_C[1:-1, 1:-1] = -AC_D
    A_E[1:-1, 1:-1] = -AE_D
    A_S[1:-1, 1:-1] = -AS_D

    # Mass flowrates for convective term
    mdot_e = (u_face[:, 1:]) * rho * delta_y / 2
    mdot_w = (u_face[:, :-1]) * rho * delta_y / 2
    mdot_n = (v_face[:-1, :]) * rho * delta_x / 2
    mdot_s = (v_face[1:, :]) * rho * delta_x / 2

    # Convective terms
    AN_c = -np.maximum(-mdot_n, 0)
    AE_c = -np.maximum(-mdot_e, 0)
    AS_c = -np.maximum(mdot_s, 0)
    AW_c = -np.maximum(mdot_w, 0)
    
    AC_c = -(AN_c + AE_c + AS_c + AW_c) + (mdot_e - mdot_w) + (mdot_n - mdot_s)

    A_N[1:-1, 1:-1] += AN_c
    A_W[1:-1, 1:-1] += AW_c
    A_C[1:-1, 1:-1] += AC_c
    A_E[1:-1, 1:-1] += AE_c
    A_S[1:-1, 1:-1] += AS_c

    return A_N, A_W, A_C, A_E, A_S

# Builds source term for NS
def Build_Source(bu, bv, b):
    
    # -(pE - pW)*delta_x/2
    bu[1:-1, 1:-1] = -0.5 * delta_x * (b[1:-1, 2:] - b[1:-1, :-2])

    # -(pN - pS)*delta_y/2
    bv[1:-1, 1:-1] = -0.5 * delta_y * (b[:-2, 1:-1] - b[2:, 1:-1])

    return bu, bv

# Builds A coefficient matrices for pressure correction terms
def Build_A_P(AP_N, AP_W, AP_C, AP_E, AP_S, A_C):

    # AP_N = delta_x^2 * (1/A_C_C + 1/A_C_N) / 2
    AP_N[1:-1, 1:-1] = (delta_x ** 2) * 0.5 * ((1 / A_C[1:-1, 1:-1]) + (1 / A_C[:-2, 1:-1]))

    # AP_W = delta_x^2 * (1/A_C_C + 1/A_C_W) / 2
    AP_W[1:-1, 1:-1] = (delta_y ** 2) * 0.5 * ((1 / A_C[1:-1, 1:-1]) + (1 / A_C[1:-1, :-2]))

    # AP_W = delta_y^2 * (1/A_C_C + 1/A_C_W) / 2
    AP_E[1:-1, 1:-1] = (delta_y ** 2) * 0.5 * ((1 / A_C[1:-1, 1:-1]) + (1 / A_C[1:-1, 2:]))

    # AP_S = delta_y^2 * (1/A_C_C + 1/A_C_S) / 2
    AP_S[1:-1, 1:-1] = (delta_x ** 2) * 0.5 * ((1 / A_C[1:-1, 1:-1]) + (1 / A_C[2:, 1:-1]))

    # AP_C = -(AP_N + AP_W + AP_E + AP_S)
    AP_C[1:-1, 1:-1] = -(AP_N[1:-1, 1:-1] + AP_W[1:-1, 1:-1] + AP_E[1:-1, 1:-1] + AP_S[1:-1, 1:-1])

    return AP_N, AP_W, AP_C, AP_E, AP_S

# Builds velocity divergence for pressure correction terms
def Build_vel_div(vel_div, u_face, v_face):

    # Div(V) = delta_y * (u_e -u_w) + delta_x * (u_n - u_s)
    vel_div[1:-1, 1:-1] = (delta_y * (u_face[:, 1:] - u_face[:, :-1]) + delta_x * (v_face[:-1, :] - v_face[1:, :]))

    return vel_div

# Correct the pressure and reinforces boundary conditions
def pressure_correction(p_star, b, p_prime, alpha):

    # p* = p^m-1 + alpha * p'
    p_star = b + p_prime * alpha

    # Reinforce boundary conditions
    p_star[0, :] = p_star[1, :]
    p_star[-1, :] = p_star[-2, :]
    p_star[:, 0] = p_star[:, 1]
    p_star[:, -1] = p_star[:, -2]

    return p_star

# Corrects velocity centroids using p'
def velocity_correction(u, v, p_prime, A_C):

    # Reinforce Boundary condition
    p_prime[0, :] = p_prime[1, :]
    p_prime[-1, :] = p_prime[-2, :]
    p_prime[:, 0] = p_prime[:, 1]
    p_prime[:, -1] = p_prime[:, -2]

    # u = -delta_y * 0.5 * (p'e - p'w)
    # negative sign is taken care of
    u[1:-1, 1:-1] += delta_y * 0.5 * (p_prime[1:-1, :-2] - p_prime[1:-1, 2:]) / A_C[1:-1, 1:-1]
    
    # v = -delta_x * 0.5 * (p'n - p's)
    # negative sign is taken care of
    v[1:-1, 1:-1] += delta_x * 0.5 * (p_prime[2:, 1:-1] - p_prime[:-2, 1:-1]) / A_C[1:-1, 1:-1]

    return u, v

# Rhie and Chow correction - unsure if this works properly, but it doesn't visually change the graphs at all
def RC_correction(u_face, v_face, A_C, b):

    # de_bar = Delta(Ve) * (1 / A_C + 1 / A_E)
    de = (delta_y * delta_x / 2) * (1 / A_C[1:-1, 1:-1] + 1 / A_C[1:-1, 2:])

    # dn_bar = Delta(Vn) * (1 / A_C + 1 / A_N)
    dn = (delta_y * delta_x / 2) * (1 / A_C[1:-1, 1:-1] + 1 / A_C[:-2, 1:-1])
    
    # u_face = u_face + (de_bar / delta_x) * [0.25 * (pE - pW + pEE - pC) + pE - pC]
    u_face[:, 1:-1] += -(de[:, 1:] / delta_x) * (((b[1:-1, 2:-1] - b[1:-1, 1:-2])) - (1/4) * (b[1:-1, 2:-1] - b[1:-1, :-3] + b[1:-1, 3:] - b[1:-1, 1:-2]))

    # u_face = u_face + (de_bar / delta_x) * [0.25 * (pN - pS + pNN - pC) + pN - pC]
    v_face[1:-1, :] += - (dn[:-1, :] / delta_y) * (((b[2:-1, 1:-1] - b[1:-2, 1:-1])) - (1/4) * (b[2:-1, 1:-1] - b[:-3, 1:-1] + b[3:, 1:-1] - b[1:-2, 1:-1]))

    return u_face, v_face

# Main SIMPLE function
def SIMPLE():

    # Define argument types for the C function
    lib.GaussSeidel.argtypes = [
    ctypes.c_int,  # x
    ctypes.c_int,  # y
    ctypes.POINTER(ctypes.c_double),  # res_u
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # AP_N
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # AP_W
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # AP_C
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # AP_E
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # AP_S
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # u
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # u_old
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # vel_div
    ctypes.c_double,  # lambda_press
    ctypes.c_int,  # max_iter_GS_p
    ctypes.c_double   # tol_GS
    ]

    # momentum coefficients
    A_N = np.ones(shape = (x, y))
    A_W = np.ones(shape = (x, y))
    A_C = np.ones(shape = (x, y))
    A_E = np.ones(shape = (x, y))
    A_S = np.ones(shape = (x, y))

    # Pressure correction coefficients
    AP_N = np.ones(shape = (x, y))
    AP_W = np.ones(shape = (x, y))
    AP_C = np.ones(shape = (x, y))
    AP_E = np.ones(shape = (x, y))
    AP_S = np.ones(shape = (x, y))

    # Terms for calculating A Coefficients
    mdot_e = np.zeros(shape = (x - 2, y - 2))
    mdot_w = np.zeros(shape = (x - 2, y - 2))
    mdot_n = np.zeros(shape = (x - 2, y - 2))
    mdot_s = np.zeros(shape = (x - 2, y - 2))

    AN_c = np.zeros(shape = (x - 2, y - 2))
    AE_c = np.zeros(shape = (x - 2, y - 2))
    AS_c = np.zeros(shape = (x - 2, y - 2))
    AW_c = np.zeros(shape = (x - 2, y - 2))
    AC_c = np.zeros(shape = (x - 2, y - 2))

    # Velocity centroids
    u = np.zeros((x, y))
    u_old = np.zeros((x, y))
    v = np.zeros((x, y))
    v_old = np.zeros((x, y))

    # Lid boundary condition
    if lid == 'top':
        u[0, 1:-1] = U_lid
    elif lid == 'bottom':
        u[-1, 1:-1] = U_lid
    elif lid == 'left':
        v[1:-1, 0] = U_lid
    elif lid == 'right':
        v[1:-1, -1] = U_lid

    # Face Velocities
    u_face = np.zeros(shape = (x - 2, y - 1)) # Shape is 32x33
    v_face = np.zeros(shape = (x - 1, y - 2)) # Shape is 33x32

    # Pressure Centroid
    p = np.zeros(shape = (x, y)) # pressure centroid
    p_star = np.zeros(shape = (x, y)) # corrected pressure
    
    # Source terms
    p_x = np.zeros(shape = (x, y)) # dp/dx
    p_y = np.zeros(shape = (x, y)) # dp/dy

    # Pressure correction terms
    p_prime = np.zeros(shape = (x, y)) # p'
    p_prime_old = np.zeros(shape = (x, y))
    vel_div = np.zeros(shape = (x, y)) # RHS of pressure correction equation

    # Residual
    res = ctypes.c_double(0.0)

    # SIMPLE loop
    for iter in range(iter_SIMPLE):

        print(f'{iter + 1}')

        u_old = u.copy()
        v_old = v.copy()
        p_prime_old = p_prime.copy()

        # Only make one set of A coefficients because thy are the same for u and v
        A_N, A_W, A_C, A_E, A_S = Build_A(A_N, A_W, A_C, A_E, A_S, u_face, v_face, mdot_n, mdot_w, mdot_e, mdot_s, AN_c, AW_c, AC_c, AE_c, AS_c)

        # Make the source terms
        p_x, p_y = Build_Source(p_x, p_y, p)
  
        # Solve for center velocities
        lib.GaussSeidel(x, y, res, A_N, A_W, A_C, A_E, A_S, u, u_old, p_x, lambda_vel, iter_GS_vel, tol_GS)
        lib.GaussSeidel(x, y, res, A_N, A_W, A_C, A_E, A_S, v, v_old, p_y, lambda_vel, iter_GS_vel, tol_GS)

        # Calculate new face velocities
        u_face[:, 1:-1] = (u[1:-1, 1:-2] + u[1:-1, 2:-1]) * 0.5 # Shape is 32x33
        v_face[1:-1, :] = (v[2:-1, 1:-1] + v[1:-2, 1:-1]) * 0.5 # Shape is 33x32

        # Rhie and Chow Correction
        u_face, v_face = RC_correction(u_face, v_face, A_C, p)
        
        # Create pressure correciton A Coefficients
        AP_N, AP_W, AP_C, AP_E, AP_S = Build_A_P(AP_N, AP_W, AP_C, AP_E, AP_S, A_C)

        # Create RHS of pressure correction equation, the divergence of velocity
        vel_div = Build_vel_div(vel_div, u_face, v_face)

        # Solve for p_prime
        lib.GaussSeidel(x, y, res, AP_N, AP_W, AP_C, AP_E, AP_S, p_prime, p_prime_old, vel_div, lambda_press, iter_GS_p, tol_GS)

        # Correct Pressure
        p_star = pressure_correction(p_star, p, p_prime, alpha)

        # Correct velocity centroids with p'
        u, v = velocity_correction(u, v, p_prime, A_C)

        # Calculate new face velocities
        u_face[:, 1:-1] = (u[1:-1, 1:-2] + u[1:-1, 2:-1]) * 0.5 # Shape is 32x33
        v_face[1:-1, :] = (v[2:-1, 1:-1] + v[1:-2, 1:-1]) * 0.5 # Shape is 33x32

        # Rhie and Chow Correction
        u_face, v_face = RC_correction(u_face, v_face, A_C, p)

    return u, v, p_star

def Plot(phi_u, phi_v):

    x1 = np.linspace(0, L, x - 2)
    y1 = np.linspace(0, L, y - 2)
    X, Y = np.meshgrid(x1, y1)
    Zu = phi_u[::-1, :][1:-1, 1:-1]
    Zv = phi_v[::-1, :][1:-1, 1:-1]
    velocity_magnitude = np.sqrt(Zu**2 + Zv**2)

   # Plot the background velocity magnitude field
    bg = plt.contourf(X, Y, velocity_magnitude, levels = 1000, cmap = 'jet')  # Smooth background
    plt.colorbar(bg, label='Velocity Magnitude')  # Add colorbar

    # Overlay quiver arrows
    #plt.quiver(X[::2, ::2], Y[::2, ::2], Zu[::2, ::2], Zv[::2, ::2], scale = 4, color='white', pivot='middle')  # Quiver arrows in white
    plt.streamplot(X, Y, Zu, Zv, color = 'white', linewidth = 1)  
    plt.title('Velocity Magnitude with Vector Arrows')
    plt.xlabel('X')
    plt.ylabel('Y')

def Plot_general(res, iters, xlabel, ylabel):

    x = np.linspace(1, iters, iters)
    y = res

    plt.plot(x,y)
    plt.grid(True)
    plt.title(f'{ylabel} vs {xlabel}')
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')

def Plot1(phi, title, Bar):

    x1 = np.linspace(0, L, x - 2)
    y1 = np.linspace(0, L, y - 2)
    X, Y = np.meshgrid(x1, y1)
    z = phi[1:-1, 1:-1][::-1, :]

    bg = plt.contourf(X, Y, z, levels = 1000, cmap = 'jet')  # Smooth background
    plt.colorbar(bg, label = f'{Bar}')  # Add colorbar
    plt.title(f'{title}')
    plt.xlabel(f'X')
    plt.ylabel(f'Y')

# Grid Size
# Global
x = 100
y = 100

# Extra space for ghost nodes
x += 2
y += 2

# Choose which Lid to move
# left, bottom, right, top
lid = 'top'

# Properties of Glycerin at 273K
# Global
rho = 1 #1276 # kg/m^3
U_lid = 1 # m/s
L = 1 # m
mu = 0.0001 # N*s/m^2

# Relaxation values
# Global
lambda_vel = 0.5
lambda_press = 1.0
alpha = 0.2

# Max iteration values
# Global
iter_GS_vel = 20
iter_GS_p = 60
iter_SIMPLE = 500

# Tolerance for GS
# Global
tol_GS = 1e-6
tol_SIMPLE = 1e-4

# Intermediate values for calculations
# Global
delta_x = L/(x-2)
delta_y = L/(y-2)

# Diffusion terms
# Global
AN_D = mu * (delta_x / delta_y)
AE_D = mu * (delta_y / delta_x)
AS_D = mu * (delta_x / delta_y)
AW_D = mu * (delta_y / delta_x)
AC_D = -(AN_D + AE_D + AS_D + AW_D)

lib = ctypes.CDLL('./GS.dll')

phi_u, phi_v, b = SIMPLE()

# Plotting all graphs of interest

# Center line values
u_center = np.zeros(shape = ((x-2)//2))
v_center = np.zeros(shape = ((y-2)//2))
u_center = phi_u[1:-1, (x-2)//2][::-1]
v_center = phi_v[(y-2)//2, 1:-1]

# Center lines
plt.figure(5)
plt.subplot(1, 2, 1)
Plot_general(u_center, np.size(u_center), xlabel = 'Distance', ylabel = 'u Velocity at x = 0.5')
plt.subplot(1, 2, 2)
Plot_general(v_center, np.size(v_center), xlabel = 'Distance', ylabel = 'x Velocity at y = 0.5')

# Velocity magnitude, u and v velocities, and pressure gradient
plt.figure(1)
plt.subplot(2, 2, 1)
Plot(phi_u, phi_v)
plt.subplot(2, 2, 2)
Plot1(phi_u, title = 'Velocity in the x-direction', Bar = 'Velocity (m/s)')
plt.subplot(2, 2, 3)
Plot1(phi_v, title = 'Velocity in the y-direction', Bar = 'Velocity (m/s)')
plt.subplot(2, 2, 4)
Plot1(b, title = 'Pressure Gradient', Bar = 'Pressure')

plt.show()
