"""
Script for generating the 3D plots of the triplet similarities.
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import Normalize

from dgs.utils.constants import PROJECT_ROOT

OUT_PATH = os.path.join(PROJECT_ROOT, "./results/plots/")

iou, oks, visual, OSNet, OSNetAIN, Resnet50, Resnet152 = np.asarray(
    [
        [10, 10, 80, 76.02, 76.72, 76.46, 74.22],
        [10, 20, 70, 77.62, 76.95, 76.75, 75.61],
        [10, 30, 60, 77.98, 77.31, 77.05, 76.74],
        [10, 40, 50, 78.95, 78.14, 77.71, 77.64],
        [10, 50, 40, 78.75, 78.17, 77.94, 77.58],
        [10, 60, 30, 77.42, 77.46, 77.14, 77.79],
        [10, 70, 20, 77.26, 77.29, 77.49, 77.12],
        [10, 80, 10, 77.48, 76.04, 77.19, 76.33],
        [20, 10, 70, 77.83, 76.53, 77.00, 76.23],
        [20, 20, 60, 77.93, 76.34, 77.44, 76.43],
        [20, 30, 50, 78.51, 77.44, 77.76, 77.92],
        [20, 40, 40, 78.27, 77.87, 78.44, 78.10],
        [20, 50, 30, 78.03, 77.57, 77.92, 78.05],
        [20, 60, 20, 78.00, 77.36, 77.24, 76.87],
        [20, 70, 10, 77.83, 77.11, 76.98, 77.32],
        [30, 10, 60, 77.50, 76.02, 77.71, 76.44],
        [30, 20, 50, 78.09, 77.24, 78.33, 78.32],
        [30, 30, 40, 78.58, 77.74, 79.26, 77.92],
        [30, 40, 30, 78.18, 77.26, 78.03, 77.71],
        [30, 50, 20, 78.15, 77.60, 77.37, 77.51],
        [30, 60, 10, 78.64, 77.11, 77.15, 77.32],
        [40, 10, 50, 78.37, 77.59, 78.66, 77.70],
        [40, 20, 40, 78.68, 77.79, 78.31, 77.62],
        [40, 30, 30, 78.16, 77.84, 78.12, 77.69],
        [40, 40, 20, 78.37, 77.46, 77.75, 77.94],
        [40, 50, 10, 78.64, 77.02, 77.38, 77.68],
        [50, 10, 40, 78.46, 77.80, 78.50, 77.64],
        [50, 20, 30, 78.21, 77.73, 77.99, 78.21],
        [50, 30, 20, 79.00, 77.67, 77.57, 78.22],
        [50, 40, 10, 78.68, 77.23, 77.54, 77.31],
        [60, 10, 30, 78.62, 78.16, 78.37, 78.16],
        [60, 20, 20, 78.60, 78.00, 77.86, 78.20],
        [60, 30, 10, 78.54, 77.02, 77.50, 76.80],
        [70, 10, 20, 79.03, 77.90, 77.69, 78.00],
        [70, 20, 10, 78.45, 77.99, 77.17, 77.06],
        [80, 10, 10, 78.18, 77.72, 77.41, 76.97],
    ],
    dtype=np.float32,
).T

modules = {"OSNet": OSNet, "OSNetAIN": OSNetAIN, "Resnet50": Resnet50, "Resnet152": Resnet152}

rows = len(modules)
cols = 3

bar_width = bar_depth = 10

fig = plt.figure(figsize=(4 * cols, 4 * rows))
fig.suptitle("Triplet Similarities on the GT Dataset", fontsize=32)

subplots: dict[str, list] = {}

outer_grid = gridspec.GridSpec(rows, 1, wspace=0.2, hspace=0.6)

for i, (name, module) in enumerate(modules.items()):
    # add row title text - Add an invisible subplot spanning the entire row
    ax = fig.add_subplot(outer_grid[i])
    ax.text(0.5, 1.2, f"Visual Module: {name}", ha="center", va="center", fontsize=20, transform=ax.transAxes)
    ax.axis("off")  # Hide the subplot

    inner_grid = gridspec.GridSpecFromSubplotSpec(1, cols, subplot_spec=outer_grid[i], wspace=0.4)

    bottom = np.ones_like(iou) * (np.min(module) - 0.5)
    bar_height = module - bottom

    norm = Normalize(vmin=min(bar_height), vmax=max(bar_height))
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm(bar_height))

    # first plot
    ax1 = fig.add_subplot(inner_grid[0], projection="3d")
    ax1.set_xlabel("% oks", rotation=50)
    ax1.set_ylabel("% iou", rotation=-10)
    ax1.set_zlabel("MOTA", rotation=90)
    ax1.view_init(25, 30, 0)
    ax1.bar3d(oks, iou, bottom, bar_width, bar_depth, bar_height, shade=True, color=colors)

    # second plot
    ax2 = fig.add_subplot(inner_grid[1], projection="3d")
    ax2.set_xlabel("% visual")
    ax2.set_ylabel("% iou", rotation=-10)
    ax2.set_zlabel("MOTA", rotation=90)
    ax2.view_init(25, 30, 0)
    ax2.bar3d(visual, iou, bottom, bar_width, bar_depth, bar_height, shade=True, color=colors)

    # third plot
    ax3 = fig.add_subplot(inner_grid[2], projection="3d")
    ax3.set_xlabel("% visual")
    ax3.set_ylabel("% oks", rotation=-10)
    ax3.set_zlabel("MOTA", rotation=90)
    ax3.view_init(25, 30, 0)
    ax3.bar3d(visual, oks, bottom, bar_width, bar_depth, bar_height, shade=True, color=colors)

    subplots[name] = [ax1, ax2, ax3]


fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

# save sub figures
for name, axes in subplots.items():
    for i, ax_i in enumerate(axes):
        extent = ax_i.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(
            os.path.join(OUT_PATH, f"triplet_similarities_GT_{name}_{i+1}.png"),
            format="png",
            bbox_inches=extent.expanded(1.4, 1.4),
        )
        fig.savefig(
            os.path.join(OUT_PATH, f"triplet_similarities_GT_{name}_{i+1}.svg"),
            format="svg",
            bbox_inches=extent.expanded(1.4, 1.4),
        )

# save figures
fig.savefig(os.path.join(OUT_PATH, "./triplet_similarities_GT.svg"), format="svg")
fig.savefig(os.path.join(OUT_PATH, "./triplet_similarities_GT.png"), format="png")

plt.show()
