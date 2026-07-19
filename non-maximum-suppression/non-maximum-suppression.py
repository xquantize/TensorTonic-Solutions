def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    n = len(boxes)

    if n == 0:
        return []

    def box_area(box):
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = box_area(box_a)
        area_b = box_area(box_b)
        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return 0.0
            
        return inter_area / union_area

    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    keep = []
    suppressed = [False] * n

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)

        for other in order:
            if other == idx or suppressed[other]:
                continue
            if iou(boxes[idx], boxes[other]) >= iou_threshold:
                suppressed[other] = True

    return keep
    