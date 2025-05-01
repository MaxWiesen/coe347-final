import numpy as np
import matplotlib.pyplot as plt

def mesh():
    x_extent = 100
    config = 'right'
    x_grid_max = 10
    theta = 5 * np.pi / 180

    # Create the figure at the start
    fig, ax = plt.subplots(figsize=(10, 5))

    x_nozz = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
          0.09, 0.099, 0.109, 0.119, 0.129, 0.139, 2.452, 3.445, 4.506,
          5.732, 7.189, 8.949, 11.096, 13.734, 17.0, 21.066, 26.161,
          32.586, 40.741, 51.167, 64.597])
    y_nozz = np.array([1.0, 1.0, 1.001, 1.002, 1.003, 1.005, 1.008, 1.01, 1.014,
          1.017, 1.021, 1.026, 1.031, 1.036, 1.042, 2.538, 3.127, 3.704,
          4.31, 4.962, 5.671, 6.441, 7.275, 8.171, 9.122, 10.109,
          11.1, 12.041, 12.84, 13.354])

    if config == 'left':
        x_nozz = x_nozz[:-1]
        y_nozz = y_nozz[:-1]
    elif config == 'right':
        x_nozz = np.append(x_nozz, 2 * x_nozz[-1] - x_nozz[-2] + 4.5)
        y_nozz = np.append(y_nozz, 13.354)

    x_scale = x_grid_max * (x_nozz.max() / x_extent) / x_nozz.max()
    x_nozz = x_nozz * x_scale
    
    # Keep y coordinates as is to preserve exact shape
    L_nozz = x_nozz[-1]
    L_open = x_grid_max - L_nozz
    
    # Ensure proper grid spacing
    sf = 100  # Scale factor for cell counts
    total_cells = 2*sf  # Total cells in x-direction
    vert1 = round(L_nozz / x_grid_max * total_cells)
    vert2 = total_cells - vert1
    cross = round(sf/2)  # Cells in y-direction
    
    # Define vertices in complete clockwise order starting from bottom left
    vertices = [
        (0, 0),                         # 0: bottom left
        (0, y_nozz[0]),                 # 1: top left
        (L_nozz, y_nozz[-1]),          # 2: top at nozzle exit
        (x_grid_max, y_nozz[-1]),       # 3: top right
        (x_grid_max, 0),                # 4: bottom right
        (L_nozz, 0),                    # 5: bottom at nozzle exit
    ]

    # Plot the nozzle shape and vertices
    ax.plot(x_nozz, y_nozz, 'b-', label='Nozzle Profile')
    ax.scatter([v[0] for v in vertices], [v[1] for v in vertices], color='red', label='Mesh Vertices')
    
    # Create and plot mesh grading
    # Nozzle block
    x_nozz_mesh = np.linspace(0, L_nozz, vert1)
    y_nozz_mesh = np.linspace(0, y_nozz[0], cross)  # Bottom to throat
    y_nozz_top = np.interp(x_nozz_mesh, x_nozz, y_nozz)  # Interpolate top surface
    
    # Opening block
    x_open_mesh = np.linspace(L_nozz, x_grid_max, vert2)
    y_open_mesh = np.linspace(0, y_nozz[-1], cross)
    
    # Plot mesh points for both blocks
    for x in x_nozz_mesh:
        y_top = np.interp(x, x_nozz, y_nozz)
        y_points = np.linspace(0, y_top, cross)
        ax.plot([x]*len(y_points), y_points, 'k.', markersize=1, alpha=0.3)
        
    for x in x_open_mesh:
        y_points = np.linspace(0, y_nozz[-1], cross)
        ax.plot([x]*len(y_points), y_points, 'k.', markersize=1, alpha=0.3)
    
    # Plot horizontal mesh lines
    for i in range(cross):
        # Nozzle block
        y_ratio = i / (cross - 1)
        y_points = y_ratio * y_nozz_top
        ax.plot(x_nozz_mesh, y_points, 'k-', linewidth=0.5, alpha=0.3)
        
        # Opening block
        y = y_open_mesh[i]
        ax.plot(x_open_mesh, [y]*len(x_open_mesh), 'k-', linewidth=0.5, alpha=0.3)
    
    ax.grid(True)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Nozzle Profile with Mesh Vertices and Grading')
    ax.legend(loc='upper left')
    ax.set_xlim(-1, x_grid_max + 1)
    plt.show()
    
    block_dict = {
    'nozzle': {
        'vertices': np.array([[0, 5, 5, 0], [1, 2, 7, 6]]),
        'cells':    (vert1, cross),
        'grading':  (1, 1)
    },
    'opening': {
        'vertices': np.array([[5, 4, 4, 5], [2, 3, 8, 7]]),
        'cells':    (vert2, cross),
        'grading':  (1, 1)
    }
}

    with open('blockMeshDict', 'w') as f:
        f.write(f'// COE 347 - FINAL PROJECT MESH ({config})')
        f.write('''\n\n
FoamFile
{
    version  2.0;
    format   ascii;
    class    dictionary;
    object   blockMeshDict;
}

convertToMeters 1.0;

vertices
(\n''')
        # Write bottom vertices
        for ind, (x, y) in enumerate(vertices[:6]):
            f.write(f'\t({x:.16e} {y * np.cos(theta / 2):.16e} {-y * np.sin(theta / 2):.16e}) // {ind}\n')
        # Write top vertices (excluding duplicates - only vertices 1-3)
        for ind, (x, y) in enumerate(vertices[1:4]):
            f.write(f'\t({x:.16e} {y * np.cos(theta / 2):.16e} {y * np.sin(theta / 2):.16e}) // {ind + 6}\n')
        f.write(''');

blocks
(''')
        for ind, (block, info) in enumerate(block_dict.items()):
            f.write(f'\t// Block {ind}\n')
            
            bottom = info['vertices'][0, :]
            top    = info['vertices'][1, :]

            # Update vertex indices to account for removed duplicates
            # For vertices > 6, subtract 5 since we removed vertices 6, 9, 10, and 11
            bottom = [b if b < 6 else b-5 for b in bottom]
            top = [t if t < 6 else t-5 for t in top]

            f.write(f"\thex ({' '.join(map(str, bottom))} {' '.join(map(str, top))}) "
                    f"({(ns := info['cells'])[0]} {ns[1]} 1) "
                    f"simpleGrading ({(gs := info['grading'])[0]} {gs[1]} 1)\n\n")
            
        f.write('''
);

edges
(
    spline 1 2
    (
''')
  # compute rotated y and z for the spline
        x_coords = x_nozz[1:-1]
        y_coords = y_nozz[1:-1]
        y_rot    = y_coords * np.cos(theta/2)
        z_lower  = -y_coords * np.sin(theta/2)
        z_upper  = y_coords * np.sin(theta/2)

        for x, y_r, z in zip(x_coords, y_rot, z_lower):
            f.write(f'\t\t({x:.8e} {y_r:.8e} {z:.8e})\n')
        f.write('\t\t)\n\n')

        # write upper spline
        f.write('\t\tspline 6 7\n\t\t(\n')
        for x, y_r, z in zip(x_coords, y_rot, z_upper):
            f.write(f'\t\t({x:.8e} {y_r:.8e} {z:.8e})\n')
        f.write('\n\t)\n);\n\n')
        with open('faces.txt', 'r') as faces:
            f.write(faces.read())
    # bottom
    # {
    #     type symmetryPlane;
    #     faces
    #     (
    #         (0 5 11 6)
    #         (5 4 10 11)
    #     );
    # }

def main():
    mesh()


if __name__ == '__main__':
    main()
