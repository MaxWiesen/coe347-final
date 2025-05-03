import numpy as np
import matplotlib.pyplot as plt
import os


##### CONFIG #####
mesh_number = 3
expansion_scale = 1.  # 1.0 is original, >1 increases expansion ratio, <1 decreases it
sf = round(np.sqrt(100000))  # Scale factor for cell counts

dir_name = f'mesh_{mesh_number}'
os.makedirs(dir_name, exist_ok=True)


def plot_mesh(x_nozz, y_nozz, x_grid_max, sf, show_reflection=False):
    """
    Plot the mesh structure including control volumes.
    
    Args:
        x_nozz: x-coordinates of nozzle profile
        y_nozz: y-coordinates of nozzle profile
        x_grid_max: maximum x coordinate of domain
        sf: scale factor for cell counts
        show_reflection: whether to show the bottom half reflection (default: False)
    """
    # Adjust figure size based on whether showing reflection
    aspect_ratio = 2 if show_reflection else 1
    fig, ax = plt.subplots(figsize=(12, 6/aspect_ratio))
    
    # Calculate mesh parameters
    L_nozz = x_nozz[-1]
    total_cells = 2*sf  # Total cells in x-direction
    vert1 = round(L_nozz / x_grid_max * total_cells)
    vert2 = total_cells - vert1
    cross = round(sf/2)  # Cells in y-direction
    
    # Generate mesh points for nozzle section
    x_nozz_mesh = np.linspace(0, L_nozz, vert1)
    y_nozz_top = np.interp(x_nozz_mesh, x_nozz, y_nozz)
    
    # Generate mesh points for extension section
    x_open_mesh = np.linspace(L_nozz, x_grid_max, vert2)
    
    # Plot nozzle profile
    ax.plot(x_nozz, y_nozz, 'b-', label='Nozzle Profile', linewidth=2)
    if show_reflection:
        ax.plot(x_nozz, -y_nozz, 'b-', linewidth=2)
    
    # Plot vertical mesh lines in nozzle section
    for x in x_nozz_mesh:
        y_top = np.interp(x, x_nozz, y_nozz)
        if show_reflection:
            ax.plot([x, x], [-y_top, y_top], 'k-', linewidth=0.5, alpha=0.3)
        else:
            ax.plot([x, x], [0, y_top], 'k-', linewidth=0.5, alpha=0.3)
    
    # Plot vertical mesh lines in extension section
    for x in x_open_mesh:
        y_top = y_nozz[-1]  # Constant height in extension
        if show_reflection:
            ax.plot([x, x], [-y_top, y_top], 'k-', linewidth=0.5, alpha=0.3)
        else:
            ax.plot([x, x], [0, y_top], 'k-', linewidth=0.5, alpha=0.3)
    
    # Plot horizontal mesh lines
    x_full = np.concatenate([x_nozz_mesh, x_open_mesh])
    for i in range(cross + 1):  # +1 to include both boundaries
        # Calculate y positions with smooth transition
        y_ratio = i / cross
        y_points = []
        for x in x_full:
            if x <= L_nozz:
                y_top = np.interp(x, x_nozz, y_nozz)
            else:
                y_top = y_nozz[-1]
            y_points.append(y_top * y_ratio)
        
        # Plot lines
        ax.plot(x_full, y_points, 'k-', linewidth=0.5, alpha=0.3)
        if show_reflection:
            ax.plot(x_full, [-y for y in y_points], 'k-', linewidth=0.5, alpha=0.3)
    
    # Add markers for key points
    throat_idx = np.argmin(y_nozz[:15])
    ax.plot(x_nozz[throat_idx], y_nozz[throat_idx], 'ro', label='Throat', markersize=8)
    if show_reflection:
        ax.plot(x_nozz[throat_idx], -y_nozz[throat_idx], 'ro', markersize=8)
    
    # Configure plot
    ax.grid(False)  # Turn off grid to avoid confusion with mesh lines
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Nozzle Mesh Structure' + (' (with reflection)' if show_reflection else ''))
    ax.legend(loc='upper right')
    
    # Set y-axis limits based on whether showing reflection
    if not show_reflection:
        ax.set_ylim(bottom=0)
    ax.axis('equal')
    
    # Add text with mesh statistics
    stats_text = f'Mesh Statistics:\n'
    stats_text += f'Nozzle section: {vert1}x{cross} cells\n'
    stats_text += f'Extension section: {vert2}x{cross} cells\n'
    stats_text += f'Total cells: {total_cells*cross}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{dir_name}/plot2.png')

def mesh():
    post_nozzle_ratio = 0.49365  # Length of extension block as ratio of nozzle length
    theta = 5 * np.pi / 180

    # Create figure with three subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Base nozzle coordinates
    x_nozz = np.array([0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.0054,
                    0.0059, 0.0065, 0.0070, 0.0907, 0.1181, 0.1433, 0.1688, 0.1958, 0.2248, 0.2563, 0.2909,
                    0.3291, 0.3715, 0.4185, 0.4710, 0.5297, 0.5956, 0.6695])
    
    y_nozz_base = np.array([0.0500, 0.0500, 0.0500, 0.0500, 0.0501, 0.0501, 0.0502, 0.0503, 0.0504, 0.0505, 0.0507,
                    0.0508, 0.0510, 0.0512, 0.0513, 0.0858, 0.0962, 0.1051, 0.1133, 0.1212, 0.1290, 0.1365,
                    0.1438, 0.1508, 0.1575, 0.1636, 0.1691, 0.1737, 0.1771, 0.1790])

    # Find throat index (minimum y value in first few points)
    throat_idx = np.argmin(y_nozz_base[:15])
    throat_y = y_nozz_base[throat_idx]
    
    # Scale y coordinates after throat while keeping throat and inlet constant
    y_nozz = y_nozz_base.copy()
    y_nozz[throat_idx:] = throat_y + (y_nozz_base[throat_idx:] - throat_y) * expansion_scale
    
    # Calculate and print expansion ratio (exit area / throat area)
    expansion_ratio = (y_nozz[-1] / throat_y) ** 2
    print(f"\nNozzle Properties:")
    print(f"Throat height: {throat_y:.6f}")
    print(f"Exit height: {y_nozz[-1]:.6f}")
    print(f"Expansion ratio (A_e/A_t): {expansion_ratio:.6f}")
    
    # Plot original and scaled profiles
    ax.plot(x_nozz, y_nozz_base, 'b--', label='Original Profile', alpha=0.5)
    ax.plot(x_nozz, y_nozz, 'r-', label=f'Scaled Profile (scale={expansion_scale:.2f})')
    
    # Add markers for throat and exit
    ax.plot(x_nozz[throat_idx], throat_y, 'ko', label='Throat')
    ax.plot(x_nozz[-1], y_nozz[-1], 'ro', label='Exit')
    
    # Configure plot
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Nozzle Profile Comparison')
    ax.legend(loc='upper left')
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{dir_name}/plot1.png')

    # Calculate domain dimensions for mesh generation
    L_nozz = x_nozz[-1]
    L_open = L_nozz * post_nozzle_ratio
    x_grid_max = L_nozz + L_open

    # After calculating mesh parameters and before writing to file
    plot_mesh(x_nozz, y_nozz, x_grid_max, sf)  # Default view (half)
    plot_mesh(x_nozz, y_nozz, x_grid_max, sf, show_reflection=True)  # Full view with reflection
    
    # Ensure proper grid spacing
    total_cells = 2*sf  # Total cells in x-direction
    vert1 = round(L_nozz / x_grid_max * total_cells)
    vert2 = total_cells - vert1
    cross = round(sf/2)  # Cells in y-direction
    
    # Define vertices in complete clockwise order starting from bottom left
    vertices = [
        (0, 0),                         # 0: bottom left
        (0, y_nozz[0]),                 # 1: top left
        (L_nozz, y_nozz[-1]),           # 2: top at nozzle exit
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
    
    ax.grid()
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Nozzle Profile with Mesh Vertices and Grading')
    ax.legend(loc='upper left')
    ax.axis('equal')
    plt.savefig(f'{dir_name}/plot3.png')
    
    block_dict = {
        'nozzle': {
            'vertices': np.array([[0, 5, 5, 0], [6, 7, 2, 1]]),
            'cells':    (vert1, cross),
            'grading':  (1, 1)
        },
        'opening': {
            'vertices': np.array([[5, 4, 4, 5], [7, 8, 3, 2]]),
            'cells':    (vert2, cross),
            'grading':  (1, 1)
        }
    }
    file_out_name = f'{dir_name}/blockMeshDict'
    with open(file_out_name, 'w') as f:
        f.write(f'// COE 347 - FINAL PROJECT MESH (expansion scale: {expansion_scale})')
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
                    f"({(ns := info['cells'])[0]} 1 {ns[1]}) "
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
    return block_dict, L_nozz, y_nozz


def single_graph(block_dict, L_nozz, y_nozz):
    with open('singleGraph.txt', 'r') as f:
        content = f.readlines()
    nozz_dx = 1 / (block_dict['nozzle']['cells'][0] + block_dict['opening']['cells'][0]) / 2
    file_out_name = f'{dir_name}/singleGraph'
    with open(file_out_name, 'w') as f:
        f.write(''.join(content).replace('x1', str(L_nozz - nozz_dx)).replace('x2', str(L_nozz + nozz_dx)).replace('y1', str(y_nozz[-1])))


def main():
    block_dict, L_nozz, y_nozz = mesh()
    single_graph(block_dict, L_nozz, y_nozz)


if __name__ == '__main__':
    main()
