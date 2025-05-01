import numpy as np
import matplotlib.pyplot as plt

def mesh():
    x_extent = 100
    config = 'right'
    x_grid_max = 3
    theta = 5 * np.pi / 180

    fig, ax = plt.subplots()
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

    # Scale coordinates to maintain proper proportions
    x_scale = x_extent / x_nozz.max()
    y_scale = 1.0  # Keep original y-scaling
    
    x_nozz = x_nozz * x_scale
    y_nozz = y_nozz * y_scale
    
    # Ensure proper grid spacing
    sf = 100  # Reduced scale factor for more manageable cell sizes
    L_nozz = x_nozz[-1]
    L_open = x_grid_max - L_nozz

    total_cells = 2*sf  # Reduced total cells for better stability
    vert1 = round(L_nozz / x_grid_max * total_cells)
    vert2 = total_cells - vert1
    cross = round(sf/2)  # Fixed cross-sectional cells for better stability
    
    block_dict = {
        'nozzle': {'vertices': np.array([0, 1, 2, 5]), 'cells': (cross, vert1), 'grading': (1, 1)},
        'opening': {'vertices': np.array([2, 3, 4, 5]), 'cells': (cross, vert2), 'grading': (1, 1)}
    }

    vertices = [
        (0, 0),                         # 0
        (0, y_nozz[0]),                 # 1
        (L_nozz, y_nozz[-1]),          # 2
        (x_grid_max, y_nozz[-1]),      # 3
        (x_grid_max, 0),               # 4
        (L_nozz, 0),                   # 5
    ]

    ax.scatter(x_nozz, y_nozz)
    ax.grid()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()

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
        for ind, (x, y) in enumerate(vertices):
            f.write(f'\t({x:.16e} {y * np.cos(theta / 2):.16e} {-y * np.sin(theta / 2):.16e}) // {ind}\n')
        for ind, (x, y) in enumerate(vertices):
            f.write(f'\t({x:.16e} {y * np.cos(theta / 2):.16e}  {y * np.sin(theta / 2):.16e}) // {ind + 6}\n')
        f.write(''');

blocks
(''')
        for ind, (block, info) in enumerate(block_dict.items()):
            f.write(f'\t// Block {ind}\n')
            verts = info["vertices"].copy()
            if ind == 0:
                verts[-2], verts[-1] = verts[-1], verts[-2]
                verts[0], verts[1] = verts[1], verts[0]
            elif ind == 1:
                verts[1], verts[-1] = verts[-1], verts[1]
            f.write(f'\thex ({str(verts)[1:-1]} {str(verts + 6)[1:-1]}) ({(ns := info["cells"])[0]} {ns[1]} 1) simpleGrading ( {(gs := info["grading"])[0]} {gs[1]} 1)\n\n')
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
        f.write('\t\tspline 7 8\n\t\t(\n')
        for x, y_r, z in zip(x_coords, y_rot, z_upper):
            f.write(f'\t\t({x:.8e} {y_r:.8e} {z:.8e})\n')
        f.write('\n\t)\n);\n\n')
        with open('faces.txt', 'r') as faces:
            f.write(faces.read())


def main():
    mesh()


if __name__ == '__main__':
    main()
