import math

def ensure_quad_points(box):
    if box is None:
        return None
    if isinstance(box, (list, tuple)):
        if len(box) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box):
            return [[int(round(pt[0])), int(round(pt[1]))] for pt in box]
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = box
            return [[int(round(x1)), int(round(y1))], [int(round(x2)), int(round(y1))], [int(round(x2)), int(round(y2))], [int(round(x1)), int(round(y2))]]
        if len(box) > 0 and isinstance(box[0], (list, tuple)):
            pts = []
            for pt in box:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pts.append([int(round(pt[0])), int(round(pt[1]))])
                if len(pts) == 4:
                    break
            if len(pts) == 4:
                return pts
    return None

def rotate_points(points, center, angle_deg):
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx, cy = center
    rotated = []
    for x, y in points:
        tx, ty = x - cx, y - cy
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        rotated.append([int(round(rx + cx)), int(round(ry + cy))])
    return rotated


