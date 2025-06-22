import numpy as np
import sympy as sp

def R_x(theta):
    theta = np.deg2rad(theta)
    return sp.Matrix([
    [1, 0, 0, 0],
    [0, sp.cos(theta), -sp.sin(theta), 0],
    [0, sp.sin(theta),  sp.cos(theta), 0],
    [0, 0, 0, 1]
    ])

def R_y(theta):
    theta = np.deg2rad(theta)
    return sp.Matrix([
    [sp.cos(theta), 0, sp.sin(theta), 0],
    [0, 1, 0, 0],
    [-sp.sin(theta), 0, sp.cos(theta), 0],
    [0, 0, 0, 1]
    ])

def R_z(theta):
    theta = np.deg2rad(theta)
    return sp.Matrix([
    [sp.cos(theta), -sp.sin(theta), 0, 0],
    [sp.sin(theta),  sp.cos(theta), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ])

def T(x_0, y_0, z_0):
    return sp.Matrix([
    [1, 0, 0, -x_0],
    [0, 1, 0, -y_0],
    [0, 0, 1,  -z_0],
    [0, 0, 0, 1]
])

def geometric_variables(theta_y, theta_z, img_size, cx=None, cy=None, cz=None):
    if(cx==None):
        cx = img_size[1]//2

    if(cy==None):
        cy = img_size[0]//2

    if(cz==None):
        cz = 0 

    Y, X = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]), indexing='ij')
    
    x, y, z = sp.symbols('x y z', real=True)
    
    v = sp.Matrix([x-cx, y-cy, z-cz, 1])
    Rz_proper = R_z(theta_z)
    Ry_fixed_translated = T(x_0 = -(img_size[0]/2 - cx), y_0=0, z_0=0) * R_y(theta_y) * T(x_0 = (img_size[0]/2 - cx), y_0=0, z_0=0)
    Ry_proper = Rz_proper * Ry_fixed_translated * Rz_proper.T
    R_total = Ry_proper * Rz_proper
    v_rot =  R_total * v
    x_r, y_r, z_r, _ = v_rot

    return cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r

def sympy_to_numpy(projection, img_size, x, y, X, Y):
    # Simplify
    projection = sp.simplify(projection)
    
    P_func = sp.lambdify((x, y), projection, modules='numpy')
    
    # --- Initialize image ---
    projection = np.zeros(img_size, dtype=np.float32)
    
    # --- Evaluate only inside the valid region ---
    with np.errstate(divide='ignore', invalid='ignore'):
        projection = P_func(X, Y)
    projection = np.nan_to_num(projection)

    return projection


def create_ellipse_proj(theta_y, theta_z, img_size, a, b, c, cx=None, cy=None, cz=None):
    
    cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r = geometric_variables(theta_y, theta_z, img_size, cx, cy, cz)
    
    # Sphere equation inside indicator function
    sphere_expr = (x_r/a)**2 + (y_r/b)**2 + (z_r/c)**2 - 1.0
    
    z_limits = sp.solve(sphere_expr, z) # z values for where sphere_expr = 0
    z_lower, z_upper = z_limits[0], z_limits[1]
    
    # Integrate 1 over z between those limits
    projection = sp.integrate(1, (z, z_lower, z_upper))
    
    projection = sympy_to_numpy(projection, img_size, x, y, X, Y)

    return projection

def create_box_proj(theta_y, theta_z, img_size, Lx, Ly, Lz, cx=None, cy=None, cz=None):

    cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r = geometric_variables(theta_y, theta_z, img_size, cx, cy, cz)
    
    # Cube indicator (1 inside cube, 0 outside)
    inside_cube = sp.Piecewise(
        (1,
         (x_r >= -Lx/2) & (x_r <= Lx/2) &
         (y_r >= -Ly/2) & (y_r <= Ly/2) &
         (z_r >= -Lz/2) & (z_r <= Lz/2)),
        (0, True)
    )
    
    # Integrate over z from -∞ to ∞ (outside cube indicator=0 so integral converges)
    projection = sp.integrate(inside_cube, (z, -sp.oo, sp.oo))
    
    projection = sympy_to_numpy(projection, img_size, x, y, X, Y)

    return projection

def create_wedge_proj(theta_y, theta_z, img_size, Lx, Ly, Lz, cx=None, cy=None, cz=None):

    cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r = geometric_variables(theta_y, theta_z, img_size, cx, cy, cz)
    
    # Sphere equation inside indicator function
    wedge_low_expr = z_r + Lz/2
    wedge_high_expr = z_r - Lz/2
    
    inside_cube = sp.Piecewise(
        (1,
        (x_r >= -Lx/2) & (x_r <= Lx/2) &
        (y_r >= -Ly/2) & (y_r <= Ly/2)),
        (0, True)
    )

    z_limits_low = sp.solve(wedge_low_expr, z) # z values for where sphere_expr = 0
    z_lower = z_limits_low[0]

    z_limits_high = sp.solve(wedge_high_expr, z) # z values for where sphere_expr = 0
    z_upper = z_limits_high[0]
    
    if(theta_y <= 90):
        projection = sp.integrate(inside_cube, (z, z_lower, -(Lz/Lx)*x_r) )
    else:
        projection = sp.integrate(inside_cube, (z, -(Lz/Lx)*x_r, z_upper) )

    projection = sympy_to_numpy(projection, img_size, x, y, X, Y)

    return projection

def create_cylinder_proj(theta_y, theta_z, img_size, D, h, cx=None, cy=None, cz=None):

    cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r = geometric_variables(theta_y, theta_z, img_size, cx, cy, cz)
    
    # Sphere equation inside indicator function
    sphere_expr = (x_r/(D/2))**2 + (y_r/(100*D))**2 + (z_r/(D/2))**2 - 1.0

    # Cube indicator (1 inside cube, 0 outside)
    inside_cube = sp.Piecewise(
        (1,
         (x_r >= -D/2) & (x_r <= D/2) &
         (y_r >= -h/2) & (y_r <= h/2) &
         (z_r >= -D/2) & (z_r <= D/2)),
        (0, True)
    )
    
    z_limits = sp.solve(sphere_expr, z) # z values for where sphere_expr = 0
    z_lower, z_upper = z_limits[0], z_limits[1]
    
    # Integrate 1 over z between those limits
    projection = sp.integrate(inside_cube, (z, z_lower, z_upper))#*y_indicator
    
    projection = sympy_to_numpy(projection, img_size, x, y, X, Y)

    return projection

def create_hollow_cylinder_proj(theta_y, theta_z, img_size, D, D_int, h, h_int, cx=None, cy=None, cz=None):

    cylinder_projection = create_cylinder_proj(theta_y=theta_y, theta_z = theta_z, img_size=img_size, D = D, h = h, cx=cx, cy=cy, cz=cz)
    cylinder_int_projection = create_cylinder_proj(theta_y=theta_y, theta_z = theta_z, img_size=img_size, D = D_int, h = h_int, cx=cx, cy=cy, cz=cz)

    hollow_cylinder_projection = cylinder_projection - cylinder_int_projection

    return hollow_cylinder_projection

def create_hollow_cube_proj(theta_y, theta_z, img_size, Lx, Ly, Lz, Lx_int, Ly_int, Lz_int, cx=None, cy=None, cz=None):

    box_projection = create_box_proj(theta_y=theta_y, theta_z = theta_z, img_size=img_size, Lx=Lx, Ly=Ly, Lz=Lz, cx=cx, cy=cy, cz=cz)
    box_int_projection = create_box_proj(theta_y=theta_y, theta_z = theta_z, img_size=img_size, Lx=Lx_int, Ly=Ly_int, Lz=Lz_int, cx=cx, cy=cy, cz=cz)

    hollow_box_projection = box_projection - box_int_projection

    return hollow_box_projection