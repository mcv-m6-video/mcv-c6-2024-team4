import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Code imported from https://github.com/mcv-m6-video/mcv-m6-2023-team2/blob/main/week4/metrics.py
def MSEN(GT, pred, output_dir, verbose=False, visualize=True):
    """
    Computes "Mean Square Error in Non-occluded areas"
    """

    u_diff, v_diff = GT[:, :, 0] - pred[:, :, 0], GT[:, :, 1] - pred[:, :, 1]
    se = np.sqrt(u_diff ** 2 + v_diff ** 2)
    sen = se[GT[:, :, 2] == 1]
    msen = np.mean(sen)

    if verbose:
        print(GT[0, -1])
        print(pred[0, -1])
        print(u_diff[0, -1])
        print(v_diff[0, -1])
        print(se[0, -1])

    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        
        se[GT[:, :, 2] == 0] = 0  # Exclude non-valid pixels
        plt.figure(figsize=(11, 4))
        img_plot = plt.imshow(se)
        img_plot.set_cmap("Blues")
        plt.title(f"Mean Square Error in Non-Occluded Areas")
        plt.colorbar()
        os.makedirs("./results", exist_ok=True)
        plt.savefig(os.path.join(output_dir, "OF_MSEN.png"))
        plt.clf()

        pred, _ = cv2.cartToPolar(pred[:, :, 0], pred[:, :, 1])
        plt.figure(figsize=(11, 4))
        img_plot = plt.imshow(pred)
        plt.clim(0,4)
        img_plot.set_cmap("YlOrRd")
        plt.title(f"Optical Flow Prediction")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, "OF_Prediction.png"))
        plt.clf()

    return msen, sen


def PEPN(sen, th=3):
    """
    Compute "Percentage of Erroneous Pixels in Non-Occluded Areas"
    """

    n_pixels_n = len(sen)
    error_count = np.sum(sen > th)
    pepn = 100 * (1 / n_pixels_n) * error_count

    return pepn