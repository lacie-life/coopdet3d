# rotate_iou_cpu_fallback.py
# Fallback CPU version for rotate IoU (boxes shape: [N,5] = [cx, cy, w, l, yaw])
import numpy as np

def _rot_rect_corners(cx, cy, w, l, yaw):
    # corners (w along x, l along y) around center
    c, s = np.cos(yaw), np.sin(yaw)
    hw, hl = 0.5 * w, 0.5 * l
    # local (x,y)
    pts = np.array([
        [ hw,  hl],
        [-hw,  hl],
        [-hw, -hl],
        [ hw, -hl],
    ], dtype=np.float32)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    rot = (pts @ R.T)
    rot[:, 0] += cx
    rot[:, 1] += cy
    return rot  # (4,2)

def _poly_area(p):
    # polygon area (shoelace), p: (M,2)
    x, y = p[:,0], p[:,1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _clip_poly(subject, clip):
    # Sutherland–Hodgman polygon clipping

    subject = np.asarray(subject, dtype=np.float32)
    clip = np.asarray(clip, dtype=np.float32)

    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0.0
    
    def intersect(a1, a2, b1, b2):
        # line a1-a2 with b1-b2
        x1,y1 = a1; x2,y2 = a2; x3,y3 = b1; x4,y4 = b2
        denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if np.abs(denom) < 1e-8:  # parallel
            return a2  # any point
        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
        return np.array([x1 + ua*(x2-x1), y1 + ua*(y2-y1)], dtype=np.float32)

    output = subject.copy()

    for i in range(len(clip)):
        input_arr = np.asarray(output, dtype=np.float32)
        if input_arr.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        A = clip[i]
        B = clip[(i + 1) % len(clip)]
        S = input_arr[-1]
        out_list = []
        for E in input_arr:
            if inside(E, A, B):
                if not inside(S, A, B):
                    out_list.append(intersect(S, E, A, B))
                out_list.append(E)
            elif inside(S, A, B):
                out_list.append(intersect(S, E, A, B))
            S = E
        output = np.asarray(out_list, dtype=np.float32)
    return np.asarray(output, dtype=np.float32)

def _pair_iou(c1, c2):
    inter_poly = _clip_poly(c1, c2)
    if inter_poly.size == 0:
        inter = 0.0
    else:
        inter = _poly_area(inter_poly)
    a1 = _poly_area(c1)
    a2 = _poly_area(c2)
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1):
    """
    CPU fallback with same signature as GPU version.
    boxes:        (N,5) [cx, cy, w, l, yaw]
    query_boxes:  (K,5) [cx, cy, w, l, yaw]
    Return: IoU matrix (N,K)
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    N, K = boxes.shape[0], query_boxes.shape[0]
    # precompute corners
    bc = [ _rot_rect_corners(b[0], b[1], b[2], b[3], b[4]) for b in boxes ]
    qc = [ _rot_rect_corners(q[0], q[1], q[2], q[3], q[4]) for q in query_boxes ]
    out = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        for j in range(K):
            out[i, j] = _pair_iou(bc[i], qc[j])
    return out
