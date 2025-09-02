import math
import time
import logging

def ensure_quad_points(box):
    t0 = time.perf_counter()
    if box is None:
        res = None
        logging.getLogger("paddleocr_app").info(f"ensure_quad_points took {(time.perf_counter()-t0)*1000:.3f} ms")
        return res
    if isinstance(box, (list, tuple)):
        if len(box) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box):
            res = [[int(round(pt[0])), int(round(pt[1]))] for pt in box]
            logging.getLogger("paddleocr_app").info(f"ensure_quad_points took {(time.perf_counter()-t0)*1000:.3f} ms")
            return res
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = box
            res = [[int(round(x1)), int(round(y1))], [int(round(x2)), int(round(y1))], [int(round(x2)), int(round(y2))], [int(round(x1)), int(round(y2))]]
            logging.getLogger("paddleocr_app").info(f"ensure_quad_points took {(time.perf_counter()-t0)*1000:.3f} ms")
            return res
        if len(box) > 0 and isinstance(box[0], (list, tuple)):
            pts = []
            for pt in box:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pts.append([int(round(pt[0])), int(round(pt[1]))])
                if len(pts) == 4:
                    break
            if len(pts) == 4:
                res = pts
                logging.getLogger("paddleocr_app").info(f"ensure_quad_points took {(time.perf_counter()-t0)*1000:.3f} ms")
                return res
    res = None
    logging.getLogger("paddleocr_app").info(f"ensure_quad_points took {(time.perf_counter()-t0)*1000:.3f} ms")
    return res

def rotate_points(points, center, angle_deg):
    t0 = time.perf_counter()
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
    logging.getLogger("paddleocr_app").info(f"rotate_points took {(time.perf_counter()-t0)*1000:.3f} ms for {len(points)} points")
    return rotated


