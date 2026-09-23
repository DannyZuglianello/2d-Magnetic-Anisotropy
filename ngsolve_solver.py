import ngsolve as ngs
from ngsolve import dx
from netgen.read_gmsh import ReadGmsh
from generate_mesh import generate_mesh
import numpy as np

MMF = 40 # A-turns
cut_error = 0 # deg
# cut_error = 5 # deg

cases = ['thick_45', 'thick_90', 'slender_45', 'slender_90']
# cases = ['thick_45']

for case in cases:
    mesh_file = f"{case}_{cut_error}deg.msh"
    if case == 'thick_45':
        leg_w = 0.100
        window_w=0.060
        window_h=0.150
        generate_mesh(window_w=window_w,
                    window_h=window_h, 
                    legrl_w=leg_w,
                    legtb_w=leg_w, 
                    beta_deg=45,
                    slack=3.0, 
                    coarsening_factor=1,
                    mesh_savefile=mesh_file)
    elif case == 'thick_90':
        leg_w = 0.100
        window_w=0.060
        window_h=0.150
        generate_mesh(window_w=window_w,
                    window_h=window_h, 
                    legrl_w=leg_w,
                    legtb_w=leg_w,  
                    beta_deg=90,
                    slack=3.0, 
                    coarsening_factor=1,
                    mesh_savefile=mesh_file)
    elif case == 'slender_45':
        leg_w = 0.050
        window_w=0.300
        window_h=0.600
        generate_mesh(window_w=window_w,
                    window_h=window_h, 
                    legrl_w=leg_w,
                    legtb_w=leg_w,   
                    beta_deg=45,
                    slack=3.0, 
                    coarsening_factor=1,
                    mesh_savefile=mesh_file)
    elif case == 'slender_90':
        leg_w = 0.050
        window_w=0.300
        window_h=0.600
        generate_mesh(window_w=window_w,
                    window_h=window_h, 
                    legrl_w=leg_w,
                    legtb_w=leg_w,  
                    beta_deg=90,
                    slack=3.0, 
                    coarsening_factor=1,
                    mesh_savefile=mesh_file)
    else:
        raise RuntimeError('Requested Case is not currently available.')
    
    mesh = ngs.Mesh(ReadGmsh(mesh_file))
    
    mu0 = 4 * np.pi * 1e-7
    nu0 = 1.0 / mu0
    nu1_core = nu0/20000
    nu2_core = nu0/1000
    
    nu1 = mesh.MaterialCF({
        'legr|legl|legt|legb': nu1_core,
        'air|coil_out|coil_in': nu0
    }, default=nu0)
    
    nu2 = mesh.MaterialCF({
        'legr|legl|legt|legb': nu2_core,
        'air|coil_out|coil_in': nu0
    }, default=nu0)
    
    theta = mesh.MaterialCF({
        'legl': np.pi / 2.0,
        'legr': np.pi / 2.0 - np.radians(cut_error),
        'legt': 0.0,
        'legb': 0.0,
        'air|coil_out|coil_in': 0.0
    }, default=0.0)
    
    c, s = ngs.cos(theta), ngs.sin(theta)
    R = ngs.CF((c, -s, 
                s,  c), dims=(2,2))
    
    Lambda = ngs.CF((nu1, 0, 
                    0, nu2), dims=(2,2))
    nu = R * Lambda * R.trans
    
    fes = ngs.H1(mesh, order=2, dirichlet="dirichlet")
    A, v = fes.TnT()
    
    def rot(u):
        return ngs.CF((ngs.Grad(u)[1], -ngs.Grad(u)[0]), dims=(2,1))
    
    a = ngs.BilinearForm(fes, symmetric=True)
    a += ngs.Trace(rot(A).trans * nu * rot(v)) * dx
    
    J0 = MMF / ngs.Integrate(1, mesh, definedon=mesh.Materials('coil_in'))
    J = mesh.MaterialCF({'coil_out': J0, 'coil_in': -J0}, default=0)
    ell = ngs.LinearForm(fes)
    ell += J * v * dx
    
    a.Assemble()
    ell.Assemble()
    
    gfA = ngs.GridFunction(fes)
    gfA.vec[:] = a.mat.Inverse(fes.FreeDofs()) * ell.vec
    
    B = rot(gfA)
    
    w_m = ngs.Trace(0.5 * B.trans * nu * B)
    
    Wm = ngs.Integrate(w_m, mesh)
    
    S = leg_w * 1.0
    L_mean = 2 * (window_w + leg_w + window_h + leg_w)
    L_inner = 2 * (window_w + window_h)
    
    Perm_mean = S / (nu1_core * L_mean)
    Perm_inner = S / (nu1_core * L_inner)
    Perm_fem = (2.0 * Wm) / (MMF**2)
    
    print(6*'#'+ f"  CASE {case.upper()}   "+6*'#')
    print(f"--- Parameters ---")
    print(f"MMF (N*I):       {MMF} A-turns")
    print(f"Current Density: {J0:.2f} A/m^2")
    print(f"Cut Error: {cut_error:.2f}°")
    print(f"\n--- Results ---")
    print(f"Magnetic Energy:        {Wm:.5f} Joules")
    print(f"Mean Path Permeance:    {1000 * Perm_mean:.5f} mWb/A-turns")
    print(f"FEM Permeance:          {1000 * Perm_fem:.5f} mWb/A-turns")
    print(f"Inner Path Permeance:   {1000 * Perm_inner:.5f} mWb/A-turns")
    print(30*'#')
    
    # --- VTK Export ---
    Bmag = ngs.Norm(B)
    
    regions_cf = mesh.MaterialCF({
        'air': 0,
        'legr': 1,
        'legl': 2,
        'legt': 3,
        'legb': 4,
        'coil_in': 5,
        'coil_out': 6,
    }, default=-1)
    
    vtk_filename = f"{case}_{cut_error}deg"
    
    vtk = ngs.VTKOutput(ma=mesh,
                        coefs=[gfA, Bmag, regions_cf],
                        names=["A", "Bmag", "Region"],
                        filename=vtk_filename,
                        subdivision=2)
    vtk.Do()
    print(f"Exported to {vtk_filename}.vtu.")