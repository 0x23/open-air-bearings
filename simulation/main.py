import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
# import scipy.sparse.linalg as splinalg
import scipy
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
import numba
import pypardiso
import time
import imageio
import os
import plot_utils

"""
Poiseuille's law describes the flow rate (Q) of a fluid through the channel in terms of various parameters 
such as the pressure drop (ΔP), viscosity (μ), channel length (L), 
and the dimensions of the channel (width: w, height: h).

1) Q = ΔP * f
2) f = (w * h^3) / (12 * μ * L)


Where:
Q is the volumetric flow rate (volume per unit time),
ΔP is the pressure drop across the channel,
w is the width of the channel,
h is the height of the channel,
μ is the dynamic viscosity of the fluid (in this case, air 1.825 x 10-5),
L is the length of the channel.
f flow coefficient

Equation 1 is similar to Ohms Law I=U*(1/R) and thus the problem can also be seen as a resistor network with
pressure being the voltage and current being the flow. 
"""


def create_border_mask(mask):
    # Perform binary dilation to get the neighbors of 'true' elements
    neighbors = scipy.ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=1)

    # Create a new mask where neighbors of 'true' elements are true
    new_mask = neighbors & (~mask)

    return new_mask


@numba.jit(nopython=True)
def build_pressure_laplacian_matrix(f, bc_mask):
    """
    build the laplacian matrix weighted by the given flow coefficients
    if the boundary condition mask is set the corresponding entry will
    be set to 1.0 and will not be influenced by its neighbours
    :param f: flow coefficients
    :param bc_mask: boundary condition mask
    :return: shape, row indices, col indices, data
    """
    h = f.shape[0]
    w = f.shape[1]

    r = np.zeros(w*h*5, dtype=np.int32)  # row
    c = np.zeros(w*h*5, dtype=np.int32)  # column
    d = np.zeros(w*h*5, dtype=np.float64)  # data
    k = 0

    def insert(row, col, data):
        nonlocal k
        r[k] = row
        c[k] = col
        d[k] = data
        k += 1

    # set inner entries of laplacian matrix
    for i in range(1, h-1):
        for j in range(1, w-1):
            index = i * w + j
            f1 = max(f[i, j - 1], f[i, j])
            f2 = max(f[i, j + 1], f[i, j])
            f3 = max(f[i - 1, j], f[i, j])
            f4 = max(f[i + 1, j], f[i, j])
            s = (f1 + f2 + f3 + f4)

            if bc_mask[i, j]:
                insert(index, index, 1.0)
            else:
                insert(index, index, -1)
                insert(index, index-1, f1 / s)
                insert(index, index+1, f2 / s)
                insert(index, index-w, f3 / s)
                insert(index, index+w, f4 / s)

    # add boundaries
    for i in range(w):
        idx = i
        insert(idx, idx, 1.0)
        idx = (h-1)*w+i
        insert(idx, idx, 1.0)

    for i in range(h):
        idx = i*w+0
        insert(idx, idx, 1.0)
        idx = i*w+(w-1)
        insert(idx, idx, 1.0)

    return (w*h, w*h), r[:k], c[:k], d[:k]


def compute_pressure_distribution(flow_coefficients, pressure_bc):
    grid_shape = flow_coefficients.shape

    # setup pressure boundary conditions
    b = pressure_bc
    bc_mask = pressure_bc > 1e-8

    # build system matrix
    # print('computing laplacian...')
    shape, r, c, d = build_pressure_laplacian_matrix(flow_coefficients, bc_mask)
    A = scipy.sparse.coo_matrix((d, (r, c)), shape=shape)

    # Solve the linear system of equations using a sparse solver
    # print('solving linear system...')
    start_time = time.time()

    # pressure = splinalg.spsolve(csr_matrix(A), b.flatten(), use_umfpack=True).reshape(grid_shape)
    pressure = pypardiso.spsolve(csr_matrix(A), b.flatten()).reshape(grid_shape)

    # print(f'solve finished in {time.time()-start_time}s')
    return pressure


def compute_relative_flow(pressure, flow_coefficients):
    """
    computes flow in mm^3/s
    :param pressure:
    :param flow_coefficients:
    :return:
    """
    flow = np.stack(np.gradient(pressure), axis=2) * flow_coefficients[:, :, np.newaxis]*10e9
    return flow


def compute_flow_coefficients(gap_height_m, dyn_viscosity):
    """
    Poiseuille's law describes the flow rate (Q) of a fluid through the channel in terms of various parameters
    such as the pressure drop (ΔP), viscosity (μ), channel length (L), and the dimensions (width: w, height: h).

    1) Q = ΔP * f
    2) f = (w * h^3) / (12 * μ * L)

    Where:
    Q is the volumetric flow rate (volume per unit time)
    ΔP is the pressure drop across the channel
    w is the width of the channel
    h is the height of the channel
    μ is the dynamic viscosity of the fluid (in this case, air 1.825 x 10-5)
    L is the length of the channel
    f flow coefficient

    :param gap_height: 2D array of gap heights
    :param grid_spacing_m: pixel gr
    :param dyn_viscosity: dynamic viscosity of the fluid in kg/(m*s)
    :return: 2D array with flow coefficients
    """
    # w and L cancel each other since both have the length of the grid spacing
    f = (gap_height_m**3) / (12.0 * dyn_viscosity)
    return f


def compute_lift_force(pressure_pa, lift_area_mask, pixel_size_m):
    """
    computes the total lift force of under the area given by the mask
    :param pressure_pa: pressre field in [Pa]
    :param lift_area_mask: mask of pixels that are to be considered
    :param pixel_size_m: pixel edge length in m
    :return:
    """
    return np.sum(pressure_pa*lift_area_mask)*(pixel_size_m**2)


def evaluate_design(design_filename, mm_per_pixel, inlet_pressure_bar):
    """
    evaluates a design using different metrics and saves results to files in the same folder as the input file
    :param design_filename: input bearing pad design image
    :param mm_per_pixel: pixel size in mm for the design
    :param inlet_pressure_bar: pressure on inlet ports
    """
    inlet_flow_coefficient = 1e-8

    # load design image
    directory, filename = os.path.split(design_filename)
    design_name, _ = os.path.splitext(filename)
    bearing_desc_img = cv2.imread(design_filename, cv2.IMREAD_COLOR)

    # generate masks and gaphight array from design image
    inlet_mask = bearing_desc_img[:, :, 1]//2 > bearing_desc_img[:, :, 0]   # green points are inlets
    inlet_borer_mask = create_border_mask(inlet_mask)
    pad_mask = bearing_desc_img[:, :, 0] != 255                             # anything not white is belongs to pad
    gap_heights_m = bearing_desc_img[:, :, 0].astype(np.float64)*1e-6       # gap heights stored in brightness
    gap_heights_m[np.logical_not(pad_mask)] = 1.0                           # set high flow for regions outside pad

    pixel_size_m = mm_per_pixel*1e-3
    grid_shape = gap_heights_m.shape

    # set boundary conditions for air inlets
    pressure_bc = np.zeros(grid_shape, dtype=np.float64)
    pressure_bc[inlet_mask] = inlet_pressure_bar*1e5                        # set inlet pressure in Pascal

    # evalute air bearing pad for different gap heights
    h = 1e-6
    heights_um = []
    lift_forces = []
    pressure_fields_bar = []

    for i in range(30):
        # compute flow coeefficients and add inlet restrictions around inlet points
        flow_coefficients = compute_flow_coefficients(gap_heights_m+h, dyn_viscosity=1.825e-5)
        flow_coefficients[inlet_borer_mask] = inlet_flow_coefficient

        # compute pressure distribution from flow coefficients
        pressure = compute_pressure_distribution(flow_coefficients, pressure_bc)
        lift_force = compute_lift_force(pressure, pad_mask, pixel_size_m)

        # store results to lists
        heights_um.append(h*1e6)
        lift_forces.append(lift_force)
        pressure_fields_bar.append(pressure*1e-5)

        # increase gap hight
        h *= 1.17
        print(f'lift force at {h*1e6:.1f}µm: {lift_force:.1f} N')

    # plot gap height animation and save to gif
    save_path = os.path.join(directory, 'pressure_'+design_name + '.gif')
    plot_utils.plot_scalar_field_animation(pressure_fields_bar,
                                           titles=[f'Pressure at h={h:.1f} µm' for h in heights_um],
                                           cbar_label='[bar]',
                                           limits=(0, inlet_pressure_bar),
                                           save_path=save_path)

    # plot summary
    h = 5*1e-6   # gap hight for summary
    flow_coefficients = compute_flow_coefficients(gap_heights_m + h, dyn_viscosity=1.825e-5)
    flow_coefficients[inlet_borer_mask] = inlet_flow_coefficient
    pressure = compute_pressure_distribution(flow_coefficients, pressure_bc)
    flow = compute_relative_flow(pressure, flow_coefficients)

    fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    ax[0, 0].set_title(f'Pressure at h={h*1e6:.1f} µm')
    plot_utils.plot_scalar_field(fig, ax[0, 0], pressure*1e-5, cbar_label='[bar]', limits=(0, inlet_pressure_bar))

    ax[0, 1].set_title(f'Relative Log Flow at h={h*1e6:.1f} µm')
    plot_utils.plot_flow_field(fig, ax[0, 1], flow, cbar_label='', limits=None)

    ax[1, 0].set_title(f'Lift force plot')
    plot_utils.makeplot_lift_force_curve(ax[1, 0], heights_um, lift_forces, (0, 40))
    # plt.show()

    # Render the plot as an image and save to file
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imwrite(os.path.join(directory, 'summary_' + design_name + '.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main():
    # specify your design here, output will be in same folder as design file
    evaluate_design('./examples/simple_single_inlet.png', mm_per_pixel=0.1, inlet_pressure_bar=0.5)
    evaluate_design('./examples/simple_multi_inlet.png', mm_per_pixel=0.1, inlet_pressure_bar=0.5)
    evaluate_design('./examples/design1h_38x18_60u.png', mm_per_pixel=0.1, inlet_pressure_bar=0.5)


if __name__ == "__main__":
    main()









# def plot_gray_image__(image, limits, gapheight_m, num_isolines=10, colormap='viridis'):
#     fig, ax = plt.subplots(dpi=150)
#
#     plt.title(f'Pressure at h={gapheight_m*1e6:.1f} µm')
#     if limits is None:
#         img = ax.imshow(image, cmap=colormap)
#     else:
#         img = ax.imshow(image, cmap=colormap, vmin=limits[0], vmax=limits[1])
#
#     if num_isolines > 0:
#         cs = ax.contour(image, num_isolines, colors='black', linewidths=0.25)
#     fig.colorbar(img, ax=ax)
#    # plt.show()
#
#     # Render the plot as an image
#     canvas = fig.canvas
#     canvas.draw()
#     image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
#
#     plt.close(fig)
#
#     return image
#
# def plot_lift_force_curve__(h, f, force_limits=None):
#     fig, ax = plt.subplots(dpi=150)
#
#     # Create a scatter plot
#     plt.plot(h, f)
#     plt.xlabel('h [µm]')
#     plt.ylabel('F [N]')
#     plt.title('Lift force plot')
#     if force_limits is not None:
#         plt.ylim(force_limits[0], force_limits[1])
#
#     # plt.show()
#
#     # Create a MultipleLocator for the subticks
#     subticks_locator = plt_ticker.MultipleLocator(base=5.0)
#     ax.xaxis.set_minor_locator(subticks_locator)
#
#     tick_interval = 10.0  # Set the interval to 0.5
#     plt.xticks(np.arange(0, max(h), tick_interval))
#
#     # Render the plot as an image
#     canvas = fig.canvas
#     canvas.draw()
#     image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
#
#     return image
#
# def make_image_grid__(images, num_columns, background_color=(255, 255, 255)):
#     num_images = len(images)
#     num_rows = int(np.ceil(num_images / num_columns))
#
#     max_height = max(image.shape[0] for image in images)
#     max_width = max(image.shape[1] for image in images)
#
#     grid_image = np.full((max_height * num_rows, max_width * num_columns, 3), background_color, dtype=np.uint8)
#
#     for i, image in enumerate(images):
#         row = i // num_columns
#         col = i % num_columns
#
#         height, width, _ = image.shape
#         start_row = row * max_height + (max_height - height) // 2
#         start_col = col * max_width + (max_width - width) // 2
#         grid_image[start_row:start_row+height, start_col:start_col+width, :] = image
#
#     return grid_image

