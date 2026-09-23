import gmsh
import numpy as np

def generate_mesh(window_w, window_h, legrl_w, legtb_w, beta_deg, slack, coarsening_factor = 1, mesh_savefile='temp.msh'):
    gmsh.initialize()
    
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
    
    gmsh.model.add('Magnetic Circuit')

    x_in = window_w / 2.0
    y_in = window_h / 2.0
    x_out = x_in + legrl_w
    y_out = y_in + legtb_w

    coil_w = (window_w / 2.0) * 0.7  
    coil_h = window_h * 0.7 
    
    max_x = x_out + coil_w
    max_y = y_out
    
    air_max_x = slack * max_x
    air_max_y = slack * max_y

    beta = np.radians(beta_deg)
    dx = np.sin(beta)
    dy = np.cos(beta)

    t_x = legrl_w / dx if dx > 1e-6 else float('inf')
    t_y = legtb_w / dy if dy > 1e-6 else float('inf')
    t = min(t_x, t_y)

    x_cut = x_in + t * dx
    y_cut = y_in + t * dy
    
    if np.isclose(x_cut, x_out, rtol = 0.01) and np.isclose(y_cut, y_out, rtol = 0.01):
        x_cut, y_cut = x_out, y_out
    
    legr_pts = [(x_in, y_in), (x_in, -y_in)]
    legt_pts = [(-x_in, y_in), (x_in, y_in)]

    if np.isclose(x_cut, x_out) and np.isclose(y_cut, y_out):
        legr_pts.extend([(x_out, -y_out), (x_out, y_out)])
        legt_pts.extend([(x_out, y_out), (-x_out, y_out)])
    elif np.isclose(x_cut, x_out):
        legr_pts.extend([(x_out, -y_cut), (x_out, y_cut)])
        legt_pts.extend([(x_out, y_cut), (x_out, y_out), (-x_out, y_out), (-x_out, y_cut)])
    else:
        legr_pts.extend([(x_cut, -y_out), (x_out, -y_out), (x_out, y_out), (x_cut, y_out)])
        legt_pts.extend([(x_cut, y_out), (-x_cut, y_out)])

    legl_pts = [(-x, y) for x, y in legr_pts][::-1]
    legb_pts = [(x, -y) for x, y in legt_pts][::-1]

    def add_poly(pts):
        p_tags = [gmsh.model.occ.addPoint(p[0], p[1], 0.0) for p in pts]
        l_tags = [gmsh.model.occ.addLine(p_tags[i], p_tags[(i+1)%len(p_tags)]) for i in range(len(p_tags))]
        cl_tag = gmsh.model.occ.addCurveLoop(l_tags)
        return gmsh.model.occ.addPlaneSurface([cl_tag])

    surfs = [add_poly(legr_pts), add_poly(legl_pts), add_poly(legt_pts), add_poly(legb_pts)]

    surfs.append(gmsh.model.occ.addRectangle(-x_out - coil_w, -coil_h / 2.0, 0, coil_w, coil_h))
    surfs.append(gmsh.model.occ.addRectangle(-x_in, -coil_h / 2.0, 0, coil_w, coil_h))
    surfs.append(gmsh.model.occ.addRectangle(-air_max_x, -air_max_y, 0, 2*air_max_x, 2*air_max_y))

    out, _ = gmsh.model.occ.fragment([(2, s) for s in surfs], [])
    gmsh.model.occ.synchronize()

    phys_surfaces = {
        'air': [], 'coil_out': [], 'coil_in': [], 
        'legr': [], 'legl': [], 'legt': [], 'legb': []
    }

    tol = 1e-5
    for dim, tag in out:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        
        if np.isclose(xmin, -x_out - coil_w, atol=tol) and np.isclose(xmax, -x_out, atol=tol):
            phys_surfaces['coil_out'].append(tag)
        elif np.isclose(xmin, -x_in, atol=tol) and np.isclose(xmax, -x_in + coil_w, atol=tol):
            phys_surfaces['coil_in'].append(tag)
        elif np.isclose(xmin, x_in, atol=tol) and np.isclose(xmax, x_out, atol=tol):
            phys_surfaces['legr'].append(tag)
        elif np.isclose(xmax, -x_in, atol=tol) and np.isclose(xmin, -x_out, atol=tol):
            phys_surfaces['legl'].append(tag)
        elif np.isclose(ymin, y_in, atol=tol) and np.isclose(ymax, y_out, atol=tol):
            phys_surfaces['legt'].append(tag)
        elif np.isclose(ymax, -y_in, atol=tol) and np.isclose(ymin, -y_out, atol=tol):
            phys_surfaces['legb'].append(tag)
        else:
            phys_surfaces['air'].append(tag)

    for name, tags in phys_surfaces.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, name=name)

    outer_bnds = gmsh.model.getBoundary(out, combined=True)
    outer_tags = [tag for dim, tag in outer_bnds]
    internal_tags = [tag for dim, tag in gmsh.model.getEntities(1) if tag not in outer_tags]
    phys_lines = {'dirichlet': [tag for _, tag in outer_bnds], 'default': internal_tags}

    for name, tags in phys_lines.items():
        if tags:
            gmsh.model.addPhysicalGroup(1, tags, name=name)
            
    size_core = legrl_w / 15.0      
    size_air = min(air_max_x, air_max_y) / 8.0    
    
    dist_min = (max(legrl_w, legtb_w) / 2.0) * 1.1 
    dist_max = max(air_max_x, air_max_y)

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", internal_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", size_core*coarsening_factor)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", size_air*coarsening_factor)
    gmsh.model.mesh.field.setNumber(2, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(2, "DistMax", dist_max)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_savefile)
    gmsh.finalize()

if __name__ == '__main__':
    leg_w = 0.100
    window_w=0.060
    window_h=0.150
    generate_mesh(window_w=window_w,
                window_h=window_h, 
                legrl_w=leg_w,
                legtb_w=leg_w, 
                beta_deg=45,
                slack=3.0, 
                coarsening_factor=1)