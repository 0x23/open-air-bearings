import numpy as np
import matplotlib.figure as mpl_figure
import matplotlib.animation as mpl_animation
import matplotlib.colorbar as mpl_colorbar
import matplotlib.collections as mpl_collections
import matplotlib.ticker as mpl_ticker
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from PIL import GifImagePlugin as GifPl
GifPl.LOADING_STRATEGY = GifPl.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY


class CustomPillowWriter(mpl_animation.AbstractMovieWriter):
    @classmethod
    def isAvailable(cls):
        return True

    def __init__(self, fps=5):
        super().__init__(fps=fps)

    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        self._frames = []

    def grab_frame(self, **savefig_kwargs):
        buf = BytesIO()
        self.fig.savefig(buf, **{**savefig_kwargs, "format": "rgba", "dpi": self.dpi})
        img = Image.frombuffer("RGBA", self.frame_size, buf.getbuffer(), "raw", "RGBA", 0, 1)
        #img = img.convert("P", palette=Image.Palette.ADAPTIVE)
        img = img.convert("RGB", palette=Image.Palette.ADAPTIVE)
        self._frames.append(img)

    def finish(self):
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0, optimize=False, lossless=True)


def plot_scalar_field(fig, ax, scalar_field, cbar_label, limits, colormap='viridis', num_isolines=10, save_path=None):
    # Add a colorbar
    if limits is not None:
        image = ax.imshow(scalar_field, cmap=colormap, vmin=limits[0], vmax=limits[1])
    else:
        image = ax.imshow(scalar_field, cmap=colormap)

    contour = ax.contour(scalar_field, num_isolines, colors='black', linewidths=0.25)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(cbar_label)


def plot_flow_field(fig, ax, flow_field, cbar_label, limits, colormap='plasma', show_arrows=False):
    gradient_magnitude = np.sqrt(np.sum(np.square(flow_field), axis=2))
    gradient_magnitude = np.log(gradient_magnitude+1e-8)

    # Add a colorbar
    if limits is not None:
        image = ax.imshow(gradient_magnitude, cmap=colormap, vmin=limits[0], vmax=limits[1])
    else:
        image = ax.imshow(gradient_magnitude, cmap=colormap)

    if show_arrows:
        # Extract the x, y, and vector components from the data
        x, y = np.meshgrid(np.arange(flow_field.shape[1]), np.arange(flow_field.shape[0]))

        # Subset the data and coordinates based on stride values
        stride = 5
        sub_data = -flow_field[::stride, ::stride, :]
        sub_x = x[::stride, ::stride]
        sub_y = y[::stride, ::stride]

        # Plot the vector field
        ax.quiver(sub_x, sub_y, sub_data[:, :, 1], sub_data[:, :, 0], angles='xy', scale=50,
                  color='white', scale_units='xy', width=0.001)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(cbar_label)
    colorbar.set_ticks([])


def plot_scalar_field_animation(scalar_fields, titles, cbar_label, limits, colormap='viridis', num_isolines=10, save_path=None):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Update function for each frame
    def update(frame):
        # Clear previous plot
        ax.clear()

        # Plot the contour of the current image
        contour = ax.contour(scalar_fields[frame], num_isolines, colors='black', linewidths=0.25)

        # Plot the false color image
        image = ax.imshow(scalar_fields[frame], cmap=colormap, vmin=limits[0], vmax=limits[1])

        # Customize the plot
        ax.set_title(titles[min(frame, len(titles)-1)])

        return [image, contour]

    # Add a colorbar
    image = ax.imshow(scalar_fields[0], cmap=colormap, vmin=limits[0], vmax=limits[1])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(cbar_label)

    # Create the animation
    animation = mpl_animation.FuncAnimation(fig, update, frames=len(scalar_fields), interval=10, blit=False)

    # Save the animation as a .gif file
    if save_path is not None:
        writer = CustomPillowWriter(fps=15)
        animation.save(save_path, writer=writer, dpi=200)  # Change the filename as needed
    else:
        plt.show()

    del animation, image, colorbar, fig


def makeplot_lift_force_curve(ax, h, f, force_limits=None):
    # Create a scatter plot
    ax.plot(h, f)
    ax.set_xlabel('h [µm]')
    ax.set_ylabel('F [N]')
    ax.set_title('Lift force plot')

    ax.grid(True)

    if force_limits is not None:
        ax.set_ylim(force_limits[0], force_limits[1])

    # Create a MultipleLocator for the subticks
    subticks_locator = mpl_ticker.MultipleLocator(base=5.0)
    ax.xaxis.set_minor_locator(subticks_locator)

    tick_interval = 10.0  # Set the interval to 0.5
    ax.set_xticks(np.arange(0, max(h), tick_interval))


