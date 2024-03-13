# compute the histogram intersection between two feature vectors
def histogram_intersection(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))

# compute the euclidian distance between two feature vectors
def euclidian_distance(hist1, hist2):
    return np.sqrt(np.sum(np.square(hist1 - hist2)))

# Code imported from https://github.com/mcv-m6-video/mcv-m6-2023-team2/blob/main/week4/utils_w4.py
def convert_optical_flow_to_image(flow):
    # The 3-channel uint16 PNG images that comprise optical flow maps contain information
    # on the u-component in the first channel, the v-component in the second channel,
    # and whether a valid ground truth optical flow value exists for a given pixel in the third channel.
    # A value of 1 in the third channel indicates the existence of a valid optical flow value
    # while a value of 0 indicates otherwise. To convert the u- and v-flow values from
    # their original uint16 format to floating point values, one can do so by subtracting 2^15 from the value,
    # converting it to float, and then dividing the result by 64.

    img_u = (flow[:, :, 2] - 2 ** 15) / 64
    img_v = (flow[:, :, 1] - 2 ** 15) / 64

    assert (flow[:, :, 0] > 1).sum() == 0 # all values are 0 or 1

    img_u[flow[:, :, 0] == 0] = 0
    img_v[flow[:, :, 0] == 0] = 0

    optical_flow = np.dstack((img_u, img_v, flow[:, :, 0]))
    return optical_flow

# by @ in https://stackoverflow.com/questions/69300562/how-to-define-the-grid-for-using-grid-search-from-scratch-in-python
def makeGrid(pars_dict):  
    keys=pars_dict.keys()
    combinations=product(*pars_dict.values())
    return [dict(zip(keys,cc)) for cc in combinations]