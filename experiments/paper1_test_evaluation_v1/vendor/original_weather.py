"""Weather functions copied unchanged from the original uploaded notebook."""
import cv2
import numpy as np

def _gaussian_mask(h, w, sigma=0.35):
    yy, xx = np.mgrid[0:h, 0:w]
    yy = (yy - h/2) / (sigma*h)
    xx = (xx - w/2) / (sigma*w)
    m = np.exp(-(xx**2 + yy**2))
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m[..., None].astype(np.float32)

def add_fog(img, strength=0.55):
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=max(3, int(0.02*max(img.shape[:2]))))
    fog  = np.clip(blur + strength*255, 0, 255).astype(np.uint8)
    mask = _gaussian_mask(img.shape[0], img.shape[1], sigma=0.38)
    out  = (img*(1-mask) + fog*mask).astype(np.uint8)
    return out

def add_rain(img, streaks=1100, alpha=0.20, angle_deg=18, length=(8,18)):
    h,w = img.shape[:2]
    overlay = img.copy()
    ang = np.deg2rad(angle_deg)
    c,s = np.cos(ang), np.sin(ang)
    for _ in range(streaks):
        L = np.random.randint(length[0], length[1]+1)
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        x2 = int(x + L*c); y2 = int(y + L*s)
        cv2.line(overlay, (x,y), (x2,y2), (255,255,255), 1, cv2.LINE_AA)
    overlay = cv2.blur(overlay, (3,3))
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

def add_snow(img, flakes=2000, alpha=0.30):
    h,w = img.shape[:2]
    overlay = img.copy()
    for _ in range(flakes):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        r = np.random.randint(1,3)
        cv2.circle(overlay, (x,y), r, (255,255,255), -1, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (3,3), 0)
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

def weather_once(img, which):
    if which == "fog":  return add_fog(img)
    if which == "rain": return add_rain(img)
    if which == "snow": return add_snow(img)
    return img
