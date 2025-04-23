import numpy as np
import matplotlib.pyplot as plt

def mesh():
    x_extent = 100
    config = 'right'
    x_grid_max = 3

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

    throat_prop = y_nozz[0] / y_nozz.max()
    length_prop = x_nozz.max() / x_extent

    x_nozz = (x_nozz - x_nozz.min()) / (x_nozz.max() - x_nozz.min()) * length_prop
    y_nozz = throat_prop + (y_nozz - y_nozz.min()) / (y_nozz.max() - y_nozz.min()) * (1 - throat_prop)

    ax.scatter(x_nozz, y_nozz)
    ax.grid()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()

    sf = 250
    vert1 = round(x_nozz[-1] / x_nozz.max() * sf * 3)
    vert2 = round((1 - x_nozz[-1] / x_grid_max) * sf)
    cross = round(y_nozz[1] / y_nozz.max() * sf * 1.5)

    block_dict = {
        'nozzle': {'vertices': np.array([0, 1, 2, 5]), 'cells': (vert1, cross), 'grading': (4, 1)},
        'opening': {'vertices': np.array([2, 3, 4, 5]), 'cells': (vert2, cross), 'grading': (1, 1)}
    }

    vertices = [
        (0, 0),                         # 0
        (0, y_nozz.min()),              # 1
        (x_nozz.max(), y_nozz.max()),   # 2
        (x_grid_max, y_nozz.max()),     # 3
        (x_grid_max, 0),                # 4
        (x_nozz.max(), 0),              # 5
    ]

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
            f.write(f'\t({x:.16e} {y:.16e} -5.0000000000000003e-02) // {ind}\n')
        for ind, (x, y) in enumerate(vertices):
            f.write(f'\t({x:.16e} {y:.16e}  5.0000000000000003e-02) // {ind + 6}\n')
        f.write(''');

blocks
(''')
        for ind, (block, info) in enumerate(block_dict.items()):
            f.write(f'\t// Block {ind}\n')
            f.write(f'\thex ({str(info["vertices"])[1:-1]} {str(info["vertices"] + 6)[1:-1]}) ({(ns := info["cells"])[0]} {ns[1]} 1) simpleGrading ( {(gs := info["grading"])[0]} {gs[1]} 1)\n\n')
        f.write('''
);

edges
(
    spline 1 2
    (
''')
        nozzle = np.array([(x_noz := x_nozz[1:-1]), y_nozz[1:-1], np.ones(len(x_noz)) * -5.00000e-02])
        f.write(str(nozzle.T)[1:-1].replace('[', '\t\t(').replace(']', ')'))
        f.write('\n\t)\n\n\tspline 7 8 \n\t(\n')
        nozzle[2, :] *= -1
        f.write(str(nozzle.T)[1:-1].replace('[', '\t\t(').replace(']', ')'))
        f.write('\n\t)\n)')

def main():
    mesh()


if __name__ == '__main__':
    main()
