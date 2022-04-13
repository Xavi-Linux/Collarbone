from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
from matplotlib.axes import Axes, Subplot
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.text import  Text

from tensorflow.keras.layers import Layer, Dense, Conv2D
from tensorflow.keras.models import Model, Sequential

from typing import Union, List, Tuple, Dict

import numpy as np

__SIZE_PER_LAYER = 15


def reshape_weights(weights: np.ndarray) -> np.ndarray:
    """
    Horizontally stack the 3rd dimension of a 3D matrix to plot in 2D
    params:
        - weights: a 3D Numpy array

    returns: a 2D Numpy array
    """

    return np.hstack([weights[:,:,i] for i in range(0, weights.shape[-1])])


def calculate_n_rows(nplots: int, ncols: int) -> int:
    nrows: int = nplots // ncols

    return nrows if nplots % ncols == 0 else nrows + 1


def calculate_matrix_index(index:int, ncols:int) -> Tuple[int, int]:
    nrow: int = index // ncols
    ncol: int = (index + ncols) % ncols

    return nrow, ncol


def get_layers(target: Union[Layer, List[Layer], Model]) -> Union[List[Layer], List[np.ndarray]]:
    """
    model case is yet to be deployed
    """

    if isinstance(target, Model):
        if isinstance(target, Sequential):

            return list(filter(lambda l: len(l.get_weights()) > 0,
                               target.layers
                               )
                        )

        else:

            extracted_weights: List[np.ndarray] = target.get_weights()

            return list(filter(lambda w: w.ndim > 1, extracted_weights))

    elif isinstance(target, Layer):

        return [target]

    return list(filter(lambda l: len(l.get_weights()) > 0,
                       target
                       )
                )


def get_horizontal_alignment(text: str) -> float:
    """
    It calculates the left-position to centre a text in a matplotlib.text.Text instance

    params:
        - text: str: text to be aligned

    returns: a float with the calculated position
    """

    return 0.5 - 0.01 * len(text)/2


def build_layout(title: str) -> Tuple[Figure, GridSpec]:
    """
    params:
        - title: str: plot's main title

    returns: a figure with 2-row grid

            +-----------------------------------------+
            |               TITLE                     |
            +-----------------------------------------+
            |                                         |
            |                                         |
            +-----------------------------------------+

    """
    fig: Figure = Figure(figsize=(20, 25), tight_layout=True)
    grid: GridSpec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[.5, 9.5], hspace=0, wspace=0)

    ax: Subplot = fig.add_subplot(grid[0, :], frameon=False)
    ax.set_axis_off()
    ax.add_patch(Rectangle(xy=(0, 0),
                           height=fig.get_figheight(), width=fig.get_figwidth(),
                           facecolor='#66cc00', edgecolor='#66cc00'
                           )
                 )

    horizontal_alignment: float = get_horizontal_alignment(title)
    ax.text(horizontal_alignment, .5, title, fontsize=20, color='#FFFFFF')

    return fig, grid


def build_subheader(layout: Figure, position: SubplotSpec, title: str) -> Figure:
    """

    """
    sup: Subplot = layout.add_subplot(position, frameon=False)
    sup.set_axis_off()
    sup.add_patch(Rectangle(xy=(0, 0),
                            height=layout.get_figheight(), width=layout.get_figwidth(),
                            facecolor='#000000', edgecolor='#000000'
                            )
                 )

    horizontal_alignment: float = get_horizontal_alignment(title)
    sup.text(horizontal_alignment, .5, title, fontsize=20, color='#FFFFFF')

    return layout


def plot_layer(layer: Union[Layer, np.ndarray], layout: Figure,
               position: SubplotSpec,
               minmax: Union[None, Tuple[float, float]]=None) -> Figure:

    weights: np.ndarray
    if isinstance(layer, Layer):
        weights = layer.get_weights()[0]

    else:
        weights = layer

    if weights.ndim == 2:
        weights: np.ndarray = np.expand_dims(weights, (-2, -1))

    elif weights.ndim == 3:
        weights: np.ndarray = np.expand_dims(weights, -1)

    if weights.ndim == 4:
        ncols: int = 4
        nrows: int = calculate_n_rows(weights.shape[-1], ncols)

        min_value: float
        max_value: float
        if not minmax:
            min_value = np.min(weights, axis=None)
            max_value = np.max(weights, axis=None)

        else:
            min_value, max_value = minmax

        colour_normaliser: Normalize = Normalize(vmin=min_value, vmax=max_value)

        subgrid: GridSpecFromSubplotSpec = position.subgridspec(nrows=nrows, ncols=ncols, hspace=.1)

        style_dict: Dict = {'xlim':(-.5, weights.shape[1] * weights.shape[2]- .5),
                            'xticks': list(range(0, weights.shape[1] * weights.shape[2])),
                            'xticklabels':['{}'.format(i) for i in range(0, weights.shape[1])]* weights.shape[2]
                            }

        for index in range(0, weights.shape[-1]):
            index: int
            subplot_row: int
            subplot_col: int

            subplot_row, subplot_col = calculate_matrix_index(index, ncols)
            reshaped_matrix: np.ndarray = reshape_weights(weights[:,:,:,index])

            ax: Subplot = layout.add_subplot(subgrid[subplot_row, subplot_col], **style_dict)
            img: AxesImage = ax.matshow(reshaped_matrix, cmap='coolwarm', norm=colour_normaliser)

            if weights.shape[2] > 1:
                for channel in range(1, weights.shape[2] + 1):
                    line: Line2D = ax.axvline(x=weights.shape[1] * channel - .5,
                                              c='#000000', linewidth=2.5
                                              )

                    text: Text = ax.text((weights.shape[1]/2 - 1) + weights.shape[1] * (channel - 1), -1.5,
                                         'Channel {}'.format(channel),
                                         fontsize=8, color='#000000'
                                         )

            for row in range(0, reshaped_matrix.shape[0]):
                for col in range(0, reshaped_matrix.shape[1]):
                    label: Text = ax.text(col - .25,row,
                                          '{0:.2f}'.format(reshaped_matrix[row, col]),
                                          fontsize=6, color='#000000')
    else:

        raise ValueError('The maximum allowed number of dimensions for a matrix is 4')

    return layout


def plot_weights(target: Union[Layer, List[Layer], Model],
                 epochs: Union[None, int]=None,
                 colour_scope: str='layer_wide') -> Figure:
    """

    """
    title: str = 'Weights after {0} epochs'.format(epochs) if epochs else 'Weights'

    layers: Union[List[Layer], List[np.ndarray]] = get_layers(target)

    fig: Figure
    grid: GridSpec
    fig, grid = build_layout(title)

    subgrid: GridSpecFromSubplotSpec = grid[1, :].subgridspec(nrows=2 * len(layers),
                                                              ncols=1,
                                                              height_ratios=[1, 9] * len(layers),
                                                              hspace=.01
                                                              )
    scope: Union[None, Tuple[float, float]] = None
    if colour_scope == 'model_wide':
        minmax: List[Tuple] = list(zip(*map(lambda l: (np.min(l.get_weights()[0], axis=None),
                                                       np.max(l.get_weights()[0], axis=None)
                                                       ) if isinstance(l, Layer) else
                                                       (np.min(l, axis=None),
                                                        np.max(l, axis=None)
                                                        ),
                                             layers
                                            )
                                       )
                                   )
        scope: Tuple[float, float] = (min(minmax[0]), max(minmax[1]))

    for i, layer in enumerate(layers):
        i: int
        layer: Layer

        fig: Figure = build_subheader(fig, subgrid[i * 2, 0], 'Layer {0}'.format(layer.name))

        fig: Figure = plot_layer(layer, fig, subgrid[i * 2 + 1, 0], minmax=scope)

    fig.set_figheight(__SIZE_PER_LAYER * len(layers))

    return fig


if __name__ == '__main__':

    l: Conv2D = Conv2D(10, input_shape=(1, 5, 5, 3), use_bias=True, kernel_size=(5, 5), padding='same')
    res = l(np.arange(75.).reshape((1, 5, 5, 3)))
    l1: Dense = Dense(3, input_shape=(5, 5), use_bias=True)
    res1 = l1(np.arange(25.).reshape((5,5)))
    f = plot_weights([l1, l, l1, l,l, l1] , epochs=100, colour_scope='layer_wide')
    f.savefig('fig')
