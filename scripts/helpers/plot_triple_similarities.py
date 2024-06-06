import matplotlib.pyplot as plt
import numpy as np

iou, oks, visual, OSNet, OSNetAIN, Resnet50, Resnet152 = np.asarray([
    [0.1, 0.1, 0.8, 76.02, 76.72, 76.46, 74.22],
    [0.1, 0.2, 0.7, 77.62, 76.95, 76.75, 75.61],
    [0.1, 0.3, 0.6, 77.98, 77.31, 77.05, 76.74],
    [0.1, 0.4, 0.5, 78.95, 78.14, 77.71, 77.64],
    [0.1, 0.5, 0.4, 78.75, 78.17, 77.94, 77.58],
    [0.1, 0.6, 0.3, 77.42, 77.46, 77.14, 77.79],
    [0.1, 0.7, 0.2, 77.26, 77.29, 77.49, 77.12],
    [0.1, 0.8, 0.1, 77.48, 76.04, 77.19, 76.33],
    [0.2, 0.1, 0.7, 77.83, 76.53, 77.00, 76.23],
    [0.2, 0.2, 0.6, 77.93, 76.34, 77.44, 76.43],
    [0.2, 0.3, 0.5, 78.51, 77.44, 77.76, 77.92],
    [0.2, 0.4, 0.4, 78.27, 77.87, 78.44, 78.10],
    [0.2, 0.5, 0.3, 78.03, 77.57, 77.92, 78.05],
    [0.2, 0.6, 0.2, 78.00, 77.36, 77.24, 76.87],
    [0.2, 0.7, 0.1, 77.83, 77.11, 76.98, 77.32],
    [0.3, 0.1, 0.6, 77.50, 76.02, 77.71, 76.44],
    [0.3, 0.2, 0.5, 78.09, 77.24, 78.33, 78.32],
    [0.3, 0.3, 0.4, 78.58, 77.74, 79.26, 77.92],
    [0.3, 0.4, 0.3, 78.18, 77.26, 78.03, 77.71],
    [0.3, 0.5, 0.2, 78.15, 77.60, 77.37, 77.51],
    [0.3, 0.6, 0.1, 78.64, 77.11, 77.15, 77.32],
    [0.4, 0.1, 0.5, 78.37, 77.59, 78.66, 77.70],
    [0.4, 0.2, 0.4, 78.68, 77.79, 78.31, 77.62],
    [0.4, 0.3, 0.3, 78.16, 77.84, 78.12, 77.69],
    [0.4, 0.4, 0.2, 78.37, 77.46, 77.75, 77.94],
    [0.4, 0.5, 0.1, 78.64, 77.02, 77.38, 77.68],
    [0.5, 0.1, 0.4, 78.46, 77.80, 78.50, 77.64],
    [0.5, 0.2, 0.3, 78.21, 77.73, 77.99, 78.21],
    [0.5, 0.3, 0.2, 79.00, 77.67, 77.57, 78.22],
    [0.5, 0.4, 0.1, 78.68, 77.23, 77.54, 77.31],
    [0.6, 0.1, 0.3, 78.62, 78.16, 78.37, 78.16],
    [0.6, 0.2, 0.2, 78.60, 78.00, 77.86, 78.20],
    [0.6, 0.3, 0.1, 78.54, 77.02, 77.50, 76.80],
    [0.7, 0.1, 0.2, 79.03, 77.90, 77.69, 78.00],
    [0.7, 0.2, 0.1, 78.45, 77.99, 77.17, 77.06],
    [0.8, 0.1, 0.1, 78.18, 77.72, 77.41, 76.97]
], dtype=np.float32).T

modules = {"OSNet": OSNet, "OSNetAIN": OSNetAIN, "Resnet50": Resnet50, "Resnet152": Resnet152}

rows = len(modules)
cols = 3

fig = plt.figure(figsize=(12, 4 * rows))

for i, (name, module) in enumerate(modules.items()):
    bottom = np.ones_like(iou) * (np.min(module) - 0.5)
    top = module - bottom
    width = depth = 0.1
    
    
    # first plot
    ax1 = fig.add_subplot(rows, cols, cols * i + 1, projection="3d")
    ax1.set_title(f"Triple Similarity - {name}")
    ax1.set_xlabel("oks")
    ax1.set_ylabel("iou")
    ax1.set_zlabel("MOTA")
    ax1.view_init(25, 30, 0)
    
    ax1.bar3d(oks, iou, bottom, width, depth, top, shade=True)
    
    # second plot
    ax2 = fig.add_subplot(rows, cols, cols * i + 2, projection="3d")
    ax2.set_title(f"Triple Similarity - {name}")
    ax2.set_xlabel("visual")
    ax2.set_ylabel("iou")
    ax2.set_zlabel("MOTA")
    ax2.view_init(25, 30, 0)
    
    ax2.bar3d(visual, iou, bottom, width, depth, top, shade=True)
    
    # third plot
    ax3 = fig.add_subplot(rows, cols, cols * i + 3, projection="3d")
    ax3.set_title(f"Triple Similarity - {name}")
    ax3.set_xlabel("visual")
    ax3.set_ylabel("oks")
    ax3.set_zlabel("MOTA")
    ax3.view_init(25, 30, 0)
    
    ax3.bar3d(visual, oks, bottom, width, depth, top, shade=True)

plt.show()

