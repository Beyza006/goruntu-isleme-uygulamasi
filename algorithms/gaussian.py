import numpy as np
import math
 
def gaussian_cekirdek_olustur(boyut, sigma):
    """
    2D Gaussian çekirdeği oluşturur.
    Formül: G(x,y) = exp(-(x²+y²) / (2σ²))
    boyut: çekirdeğin kenar uzunluğu (tek sayı olmalı, ör: 3, 5, 7)
    sigma: Gaussian dağılımının standart sapması (ne kadar büyükse o kadar bulanık)
    """
    merkez = boyut // 2
    cekirdek = np.zeros((boyut, boyut), dtype=np.float64)
 
    for i in range(boyut):
        for j in range(boyut):
            x = i - merkez
            y = j - merkez
            cekirdek[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
 
    # Toplamı 1'e normalize et (parlaklık korunur)
    cekirdek /= cekirdek.sum()
    return cekirdek
 
 
def konvolüsyon_uygula(kanal, cekirdek):
    """
    Tek kanallı (2D) görüntüye konvolüsyon uygular.
    Kenar piksellerinde sıfır doldurma (zero-padding) kullanılır.
    """
    h, w = kanal.shape
    ch, cw = cekirdek.shape
    pad_h, pad_w = ch // 2, cw // 2
 
    # Kenarları sıfırla doldur
    dolgulu = np.pad(kanal, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
 
    sonuc = np.zeros_like(kanal, dtype=np.float64)
 
    for i in range(h):
        for j in range(w):
            # Çekirdekle örtüşen bölgeyi al ve nokta çarpımını hesapla
            bolge = dolgulu[i:i + ch, j:j + cw]
            sonuc[i, j] = np.sum(bolge * cekirdek)
 
    return np.clip(sonuc, 0, 255).astype(np.uint8)
 
 
def gaussian_blur(img, cekirdek_boyutu=5, sigma=1.0):
    """
    Görüntüye Gaussian bulanıklaştırma uygular.
    cekirdek_boyutu: büyüdükçe daha çok bulanıklaştırır (tek sayı olmalı)
    sigma: Gaussian standart sapması
    """
    if cekirdek_boyutu % 2 == 0:
        raise ValueError("Çekirdek boyutu tek sayı olmalıdır (3, 5, 7, ...).")
 
    cekirdek = gaussian_cekirdek_olustur(cekirdek_boyutu, sigma)
 
    if len(img.shape) == 2:
        # Gri görüntü
        return konvolüsyon_uygula(img, cekirdek)
    else:
        # Renkli görüntü: her kanalı ayrı işle
        kanallar = [konvolüsyon_uygula(img[:, :, c], cekirdek) for c in range(img.shape[2])]
        return np.stack(kanallar, axis=2)