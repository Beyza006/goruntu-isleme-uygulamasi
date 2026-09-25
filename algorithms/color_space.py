import numpy as np
 
# --- Her fonksiyon BGR girdi alır (OpenCV standardı) ---
 
def bgr_to_rgb(img):
    """BGR → RGB: kanal sırasını ters çevir."""
    return img[..., ::-1].copy()
 
 
def bgr_to_hsv(img):
    """
    BGR → HSV dönüşümü.
    H: ton (0-360), S: doygunluk (0-1), V: parlaklık (0-1)
    Çıktı uint8 formatında döner (görüntüleme için ölçeklendirilir).
    """
    img_f = img.astype(np.float32) / 255.0
    b, g, r = img_f[..., 0], img_f[..., 1], img_f[..., 2]
 
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin
 
    # Ton hesabı
    h = np.zeros_like(cmax)
    mask = delta != 0
    h[mask & (cmax == r)] = ((g - b)[mask & (cmax == r)] / delta[mask & (cmax == r)]) % 6
    h[mask & (cmax == g)] = ((b - r)[mask & (cmax == g)] / delta[mask & (cmax == g)]) + 2
    h[mask & (cmax == b)] = ((r - g)[mask & (cmax == b)] / delta[mask & (cmax == b)]) + 4
    h = (h * 60) % 360
    h[h < 0] += 360
 
    # Doygunluk ve parlaklık
    s = np.zeros_like(cmax)
    s[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
    v = cmax
 
    hsv = np.stack([h / 360.0, s, v], axis=-1)
    return (hsv * 255).astype(np.uint8)
 
 
def bgr_to_ycrcb(img):
    """
    BGR → YCrCb dönüşümü.
    Y: parlaklık, Cr ve Cb: renk farkı bileşenleri.
    """
    img_f = img.astype(np.float32)
    b, g, r = img_f[..., 0], img_f[..., 1], img_f[..., 2]
 
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
 
    return np.clip(np.stack([y, cr, cb], axis=-1), 0, 255).astype(np.uint8)
 
 
def bgr_to_lab(img):
    """
    BGR → LAB dönüşümü (D65 ışık kaynağı referans alınır).
    L: aydınlık, A ve B: renk eksenleri.
    """
    img_f = img.astype(np.float32) / 255.0
    b, g, r = img_f[..., 0], img_f[..., 1], img_f[..., 2]
 
    # Önce XYZ'ye çevir
    X = (r * 0.412453 + g * 0.357580 + b * 0.180423) / 0.950456
    Y = (r * 0.212671 + g * 0.715160 + b * 0.072169)
    Z = (r * 0.019334 + g * 0.119193 + b * 0.950227) / 1.088754
 
    # f(t) fonksiyonu: doğrusal/kübik geçiş
    def f(t):
        return np.where(t > 0.008856, t ** (1 / 3), (903.3 * t + 16) / 116)
 
    L = 116 * f(Y) - 16
    A = 500 * (f(X) - f(Y))
    B = 200 * (f(Y) - f(Z))
 
    lab = np.stack([L, A, B], axis=-1)
    return np.clip(lab * 255 / np.max(lab), 0, 255).astype(np.uint8)
 
 
def bgr_to_xyz(img):
    """BGR → XYZ renk uzayı dönüşümü."""
    img_f = img.astype(np.float32)
    b, g, r = img_f[..., 0], img_f[..., 1], img_f[..., 2]
 
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
 
    return np.clip(np.stack([X, Y, Z], axis=-1), 0, 255).astype(np.uint8)
 
 
def bgr_to_hls(img):
    """BGR → HLS (Ton, Aydınlık, Doygunluk) dönüşümü."""
    img_f = img.astype(np.float32) / 255.0
    b, g, r = img_f[..., 0], img_f[..., 1], img_f[..., 2]
 
    cmax = np.max(img_f, axis=-1)
    cmin = np.min(img_f, axis=-1)
    delta = cmax - cmin
 
    l = (cmax + cmin) / 2
 
    s = np.zeros_like(l)
    s[delta != 0] = delta[delta != 0] / (1 - np.abs(2 * l[delta != 0] - 1))
 
    h = np.zeros_like(l)
    mask = delta != 0
    h[mask & (cmax == r)] = ((g - b)[mask & (cmax == r)] / delta[mask & (cmax == r)]) % 6
    h[mask & (cmax == g)] = ((b - r)[mask & (cmax == g)] / delta[mask & (cmax == g)]) + 2
    h[mask & (cmax == b)] = ((r - g)[mask & (cmax == b)] / delta[mask & (cmax == b)]) + 4
    h = (h / 6) * 360
 
    hls = np.stack([h / 360.0, l, s], axis=-1)
    return (hls * 255).astype(np.uint8)
 
 
def bgr_to_yuv(img):
    """BGR → YUV dönüşümü. Y: parlaklık, U ve V: renk farkı."""
    img_f = img.astype(np.float32)
    b, g, r = img_f[..., 0], img_f[..., 1], img_f[..., 2]
 
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128
    v =  0.615   * r - 0.51499 * g - 0.10001 * b + 128
 
    return np.clip(np.stack([y, u, v], axis=-1), 0, 255).astype(np.uint8)
 
 
def bgr_to_luv(img):
    """BGR → LUV dönüşümü (algısal renk uzayı)."""
    img_f = img.astype(np.float32) / 255.0
    b, g, r = img_f[..., 0], img_f[..., 1], img_f[..., 2]
 
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
 
    L = np.where(Y > 0.008856, 116 * Y ** (1 / 3) - 16, 903.3 * Y)
 
    denom = X + 15 * Y + 3 * Z + 1e-6
    u_prime = (4 * X) / denom
    v_prime = (9 * Y) / denom
 
    # D65 referans beyaz noktası için sabitler
    un, vn = 0.1978398, 0.4683363
 
    U = 13 * L * (u_prime - un)
    V = 13 * L * (v_prime - vn)
 
    luv = np.stack([L, U, V], axis=-1)
    return np.clip(luv, 0, 255).astype(np.uint8)