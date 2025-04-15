import zipfile
import xml.etree.ElementTree as ET
import sympy as sp
import numpy as np
import math

with zipfile.ZipFile("EX3.ggb", 'r') as zip_ref:
    zip_ref.extractall("extracted_ggb")

tree = ET.parse('extracted_ggb/geogebra.xml')
root = tree.getroot()

# Extract points
points = {}
for element in root.findall(".//element[@type='point']"):
    label = element.get("label")
    coords = element.find("coords")
    x = float(coords.get("x"))
    y = float(coords.get("y"))
    points[label] = {"x": x, "y": y}

# Extract segments
segments = []
for command in root.findall(".//command[@name='Segment']"):
    input_points = command.find("input")
    start_point = input_points.get("a0")
    end_point = input_points.get("a1")
    segments.append({"start": start_point, "end": end_point})

# Extract visible force vectors
forces = []
for element in root.findall(".//element[@type='vector']"):
    label = element.get("label")
    coords = element.find("coords")
    x = float(coords.get("x"))
    y = float(coords.get("y"))
    start_point_elem = element.find("startPoint")
    start_point = start_point_elem.get("exp") if start_point_elem is not None else None
    forces.append({"label": label, "x": x, "y": y, "start": start_point})

# Define variables for segment tensions
tensions = {seg["start"] + seg["end"]: sp.symbols(f"T_{seg['start']}{seg['end']}") for seg in segments}

# Identifying PIN and ROLL joints
pin_joints = [label for label in points if label.startswith("PIN")]
roll_joints = [label for label in points if label.startswith("ROLL")]

# Ground reaction forces for PIN and ROLL joints
ground_forces = {}
for pin in pin_joints:
    ground_forces[f"F_{pin}_x"] = sp.symbols(f"F_{pin}_x")
    ground_forces[f"F_{pin}_y"] = sp.symbols(f"F_{pin}_y")

for roll in roll_joints:
    ground_forces[f"F_{roll}_y"] = sp.symbols(f"F_{roll}_y")  # Only Y-direction forces for ROLL joints

equations_x = {}
equations_y = {}

for point_label, point_coords in points.items():
    eq_x, eq_y = 0, 0

    for seg in segments:
        if seg["start"] == point_label or seg["end"] == point_label:
            other_point = seg["end"] if seg["start"] == point_label else seg["start"]
            dx = points[other_point]["x"] - point_coords["x"]
            dy = points[other_point]["y"] - point_coords["y"]
            length = sp.sqrt(dx**2 + dy**2)
            tension_var = tensions[seg["start"] + seg["end"]] if seg["start"] + seg["end"] in tensions else tensions[seg["end"] + seg["start"]]
            eq_x += tension_var * (dx / length)
            eq_y += tension_var * (dy / length)

    for force in forces:
        if force["start"] == point_label:
            eq_x += 1000 * force["x"]
            eq_y += 1000 * force["y"]

    # Ground reaction forces at PIN joints (both X and Y directions)
    if point_label.startswith("PIN"):
        eq_x += ground_forces[f"F_{point_label}_x"]
        eq_y += ground_forces[f"F_{point_label}_y"]

    # Ground reaction forces at ROLL joints (only Y direction)
    if point_label.startswith("ROLL"):
        eq_y += ground_forces[f"F_{point_label}_y"]

    equations_x[point_label] = eq_x
    equations_y[point_label] = eq_y

# System of linear equations
all_equations = list(equations_x.values()) + list(equations_y.values())
variables_to_solve = list(tensions.values()) + list(ground_forces.values())

# Solving the system of equations
solutions = sp.solve(all_equations, variables_to_solve)

# Forces in the members
print("\nForces in member (+ : Tension, - : compression):")
print(f"{'Member':<20}{'Force (N)':>20}")
print("-" * 40)
for segment, tension_var in tensions.items():
    force_val = solutions.get(tension_var, None)
    if force_val is not None:
        print(f"{segment:<20}{float(force_val):>20.3f} N")
    else:
        print(f"{segment:<20}{'Not Solved':>20}")

# Ground Reaction Forces
print("\nGround Reaction Forces:")
print(f"{'Force':<20}{'Value (N)':>20}")
print("-" * 40)
for force_var_name, force_var_symbol in ground_forces.items():
    value = solutions.get(force_var_symbol, None)
    if value is not None:
        print(f"{force_var_name:<20}{float(value):>20.3f} N")
    else:
        print(f"{force_var_name:<20}{'Not Solved':>20}")



# material properties
E = float(input("\nEnter Young's Modulus of the material (in GPa): ")) * 1e9
A = float(input("Enter the cross-sectional area of each member (in m²): "))

# Number of DOFs: 2 per point
dof_map = {}
index = 0
for label in points:
    dof_map[label] = (index, index + 1)
    index += 2

num_dof = len(points) * 2
K_global = np.zeros((num_dof, num_dof))
F_global = np.zeros(num_dof)

#global stiffness matrix
for seg in segments:
    start, end = seg["start"], seg["end"]
    x1, y1 = points[start]["x"], points[start]["y"]
    x2, y2 = points[end]["x"], points[end]["y"]
    dx, dy = x2 - x1, y2 - y1
    L = (dx**2 + dy**2)**0.5
    c = dx / L
    s = dy / L

    k_local = (A * E / L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])

    dofs = dof_map[start] + dof_map[end]
    for i in range(4):
        for j in range(4):
            K_global[dofs[i], dofs[j]] += k_local[i, j]

for force in forces:
    if force["start"] in dof_map:
        dof_x, dof_y = dof_map[force["start"]]
        F_global[dof_x] += 1000 * force["x"]  # kN to N
        F_global[dof_y] += 1000 * force["y"]

# Apply boundary conditions (fixed DOFs at supports)
fixed_dofs = []
for joint in pin_joints:
    fixed_dofs += list(dof_map[joint])
for joint in roll_joints:
    fixed_dofs.append(dof_map[joint][1])  

all_dofs = set(range(num_dof))
free_dofs = list(all_dofs - set(fixed_dofs))

K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
F_reduced = F_global[free_dofs]

U_reduced = np.linalg.solve(K_reduced, F_reduced)

U = np.zeros(num_dof)
U[free_dofs] = U_reduced

# Node Displacements in mm
print("\nNode Displacements (in millimeters):")
print(f"{'Node':<20}{'Δx (mm)':>20}{'Δy (mm)':>20}")
print("-" * 60)
for point, (ix, iy) in dof_map.items():
    ux, uy = U[ix], U[iy]
    # Convert to millimeters by multiplying by 1000
    print(f"{point:<20}{ux*1000:>20.3f}{uy*1000:>20.3f}")

print("\nStress and Strain in Members:")
print(f"{'Member':<20} {'Stress (Pa)':>20} {'Strain':>20}")
print("-" * 60)

for seg in segments:
    seg_key = seg["start"] + seg["end"]
    if seg_key not in tensions:
        seg_key = seg["end"] + seg["start"]

    tension = solutions.get(tensions[seg_key], None)

    if tension is not None:
        tension = float(tension)
        stress = tension / A
        strain = stress / E
        print(f"{seg_key:<20} {stress:>20.3e} {strain:>20.3e}")
    else:
        print(f"{seg_key:<20} {'Not Solved':>40}")

yield_strength_MPa = float(input("\nEnter the yield strength of the material (in MPa): "))
yield_strength = yield_strength_MPa * 1e6  # Convert to Pascals

I = A**2 / (4 * math.pi)
r = (I / A) ** 0.5  
k = 1  

max_tensile_stress = 0
max_tension_member = None

max_compressive_stress = 0
max_compression_member = None
max_compression_length = 0

for seg in segments:
    seg_key = seg["start"] + seg["end"]
    if seg_key not in tensions:
        seg_key = seg["end"] + seg["start"]

    force = solutions.get(tensions.get(seg_key), None)

    if force is None:
        continue

    force = float(force)  
    stress = force / A  #

    if stress > 0:  
        if stress > max_tensile_stress:
            max_tensile_stress = stress
            max_tension_member = seg_key
    else: 
        compressive_stress = abs(stress)  
        if compressive_stress > max_compressive_stress:
            max_compressive_stress = compressive_stress
            max_compression_member = seg_key
            dx = points[seg["end"]]["x"] - points[seg["start"]]["x"]
            dy = points[seg["end"]]["y"] - points[seg["start"]]["y"]
            length = sp.sqrt(dx**2 + dy**2)
            max_compression_length = length  

# Euler critical stress for the most critical compressive member
sigma_cr = (math.pi ** 2 * E) / ((k * max_compression_length / r) ** 2) if max_compression_member else None

print("\nMaximum Stress Check:")

if max_tension_member:
    status_tension = "FAIL" if max_tensile_stress > yield_strength else "SAFE"
    print(f"Max Tension: Member {max_tension_member} → {max_tensile_stress:.3e} Pa vs Yield Strength {yield_strength:.3e} Pa → {status_tension}")

if max_compression_member:
    if sigma_cr:
        status_compression = "FAIL" if max_compressive_stress > sigma_cr else "SAFE"
        print(f"Max Compression: Member {max_compression_member} → {max_compressive_stress:.3e} Pa vs Euler Buckling Stress {sigma_cr:.3e} Pa → {status_compression}")
    else:
        print(f"Max Compression: No compression member found.")




