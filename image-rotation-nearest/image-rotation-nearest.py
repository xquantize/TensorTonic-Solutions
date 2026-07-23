def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    H = len(image)
    W = len(image[0])

    cy = (H - 1) / 2
    cx = (W - 1) / 2

    theta = math.radians(angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    output = []

    for i in range(H):
        row = []

        for j in range(W):
            dy = i - cy
            dx = j - cx

            src_y = cy + dy * cos_t + dx * sin_t
            src_x = cx - dy * sin_t + dx * cos_t

            src_i = round(src_y)
            src_j = round(src_x)

            if 0 <= src_i < H and 0 <= src_j < W:
                row.append(image[src_i][src_j])
            else:
                row.append(0)

        output.append(row)

    return output
