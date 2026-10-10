import argparse, os, sys
import cv2
import numpy as np

def get_markers(img):
    dct = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    try:
        det = cv2.aruco.ArucoDetector(dct, cv2.aruco.DetectorParameters())
        crn, ids, _ = det.detectMarkers(img)
    except AttributeError:
        crn, ids, _ = cv2.aruco.detectMarkers(img, dct, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None: sys.exit("No markers detected")
    ids = ids.ravel()
    found = {int(ids[i]): crn[i][0] for i in range(len(ids))}
    if any(x not in found for x in [80, 85, 90, 95]): sys.exit("Missing required markers")
    return found

def order_pts(pts_dict):
    pts = np.array(list(pts_dict.values()), dtype=np.float32)
    s, d = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)

def warp(img, mkrs):
    ctr = np.mean([p.mean(0) for p in mkrs.values()], axis=0)
    inner = {k: p[np.argmin(((p - ctr)**2).sum(1))] for k, p in mkrs.items()}
    src = order_pts(inner)
    dst = np.float32([[0, 0], [900, 0], [900, 900], [0, 900]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (900, 900))

def get_blobs(hsv, lower, upper, l2=None, u2=None):
    m = cv2.inRange(hsv, np.array(lower), np.array(upper))
    if l2 is not None:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, np.array(l2), np.array(u2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2), cv2.MORPH_OPEN, k)
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in c if cv2.contourArea(cnt) > 300]

def closest(c, pts):
    m = cv2.moments(c)
    if m['m00']:
        cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
    else:
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w/2, y + h/2
    return min(pts.keys(), key=lambda k: (cx - pts[k][0])**2 + (cy - pts[k][1])**2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    img_path = ap.parse_args().image
    
    img = cv2.imread(img_path)
    if img is None: sys.exit("Invalid image path")

    mkrs = get_markers(img)
    arena = warp(img, mkrs)
    c_px = 900 / 12
    grid = {chr(65+c) + str(r+1): (c_px*(c+1), c_px*(r+1)) for c in range(11) for r in range(11)}
    hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
    r_c = get_blobs(hsv, [0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255])
    y_c = get_blobs(hsv, [20, 100, 100], [35, 255, 255])
    crit = sorted({closest(c, grid) for c in r_c})
    stab = sorted({closest(c, grid) for c in y_c})
    ids_out = sorted(mkrs.keys())
    out_file = os.path.splitext(img_path)[0] + '_results.txt'
    with open(out_file, 'w') as f:
        f.write(f"Detected marker IDs: {ids_out}\n")
        f.write(f"Critical Survivors: {', '.join(crit)}\n")
        f.write(f"Stable Survivors: {', '.join(stab)}\n")

if __name__ == '__main__':
    main()
