import numpy as np
import matplotlib.pyplot as plt
 
# ── Gri görüntü üzerinde histogram eşitleme ──────────────────────────────────
 
def histogram_esitle_gri(img):
    """
    Gri görüntüde histogram eşitleme uygular.
    CDF (kümülatif dağılım fonksiyonu) kullanılarak piksel değerleri 0-255'e yayılır.
    """
    satirlar, sutunlar = img.shape
    toplam_piksel = satirlar * sutunlar
 
    # 1) Her gri değerin frekansını say
    hist = [0] * 256
    for i in range(satirlar):
        for j in range(sutunlar):
            hist[img[i, j]] += 1
 
    # 2) CDF hesapla (kümülatif toplam)
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
 
    # 3) Sıfır olmayan ilk CDF değerini bul (normalizasyon için)
    cdf_min = next(v for v in cdf if v > 0)
 
    # 4) Her gri değer için yeni piksel değerini hesapla
    cdf_norm = [0] * 256
    for i in range(256):
        cdf_norm[i] = round((cdf[i] - cdf_min) * 255 / (toplam_piksel - cdf_min))
 
    # 5) Eşitlenmiş görüntüyü oluştur
    esitlenmis = np.zeros_like(img)
    for i in range(satirlar):
        for j in range(sutunlar):
            esitlenmis[i, j] = cdf_norm[img[i, j]]
 
    return esitlenmis
 
 
# ── Renkli görüntü üzerinde histogram germe (YCrCb uzayında) ─────────────────
 
def histogram_gerdirme_renkli(img):
    """
    Renkli görüntüde histogram germe uygular.
    Yalnızca parlaklık kanalı (Y) üzerinde işlem yapılır,
    renk bilgisi (Cr, Cb) korunur. Bu sayede renkler bozulmaz.
    """
    satirlar, sutunlar, _ = img.shape
    img_f = img.astype(np.float32)
    b = img_f[:, :, 0]
    g = img_f[:, :, 1]
    r = img_f[:, :, 2]
 
    # BGR → YCrCb dönüşümü (manuel)
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128 + 0.5   * r - 0.419 * g - 0.081 * b
    cb = 128 - 0.169 * r - 0.331 * g + 0.5   * b
 
    # Y kanalını uint8'e dönüştür
    y_uint8 = np.clip(np.round(y), 0, 255).astype(np.uint8)
 
    # Y kanalı için histogram ve CDF
    hist = [0] * 256
    for i in range(satirlar):
        for j in range(sutunlar):
            hist[y_uint8[i, j]] += 1
 
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
 
    cdf_min = next(v for v in cdf if v > 0)
    toplam = satirlar * sutunlar
 
    # Yeni Y değerlerini hesapla (100-200 aralığına ger)
    eslem = [0] * 256
    for i in range(256):
        if cdf[i] == 0:
            eslem[i] = 0
        else:
            eslem[i] = max(100, min(200, round((cdf[i] - cdf_min) * (200 - 100) / (toplam - cdf_min) + 100)))
 
    y_yeni = np.array([[eslem[y_uint8[i, j]] for j in range(sutunlar)] for i in range(satirlar)], dtype=np.float32)
 
    # YCrCb → BGR geri dönüşümü
    sonuc = np.zeros_like(img)
    for i in range(satirlar):
        for j in range(sutunlar):
            yv = float(y_yeni[i, j])
            crv = cr[i, j]
            cbv = cb[i, j]
 
            rv = yv + 1.403 * (crv - 128)
            gv = yv - 0.714 * (crv - 128) - 0.344 * (cbv - 128)
            bv = yv + 1.773 * (cbv - 128)
 
            sonuc[i, j, 0] = max(0, min(255, round(bv)))
            sonuc[i, j, 1] = max(0, min(255, round(gv)))
            sonuc[i, j, 2] = max(0, min(255, round(rv)))
 
    return sonuc
 
 
# ── Histogram grafiği çizimi ─────────────────────────────────────────────────
 
def histogram_goster(orijinal, islenmis):
    """Orijinal ve işlenmiş görüntünün histogramlarını yan yana çizer."""
    fig, eksenler = plt.subplots(2, 1, figsize=(6, 4))
 
    for idx, (goruntu, baslik) in enumerate([(orijinal, "Orijinal"), (islenmis, "İşlenmiş")]):
        if len(goruntu.shape) == 2:
            eksenler[idx].hist(goruntu.ravel(), bins=256, color='gray')
        else:
            for kanal, renk in enumerate(('b', 'g', 'r')):
                eksenler[idx].hist(goruntu[:, :, kanal].ravel(), bins=256, color=renk, alpha=0.5)
        eksenler[idx].set_title(f"{baslik} Görüntü Histogramı")
 
    plt.tight_layout()
    return fig