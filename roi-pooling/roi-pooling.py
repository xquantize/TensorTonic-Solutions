import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    results = []

    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_h = y2 - y1
        roi_w = x2 - x1

        grid = []

        for i in range(output_size):
            row = []

            for j in range(output_size):
                h_start = y1 + (i * roi_h) // output_size
                h_end = y1 + ((i + 1) * roi_h) // output_size

                w_start = x1 + (j * roi_w) // output_size
                w_end = x1 + ((j + 1) * roi_w) // output_size

                if h_end == h_start:
                    h_end = h_start + 1
                if w_end == w_start:
                    w_end = w_start + 1

                bin_max = max(
                    feature_map[r][c]
                    for r in range(h_start, h_end)
                    for c in range(w_start, w_end)
                )
                row.append(bin_max)

            grid.append(row)

        results.append(grid)

    return results
