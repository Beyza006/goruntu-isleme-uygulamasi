import numpy as np
 
def _konvolüsyon(kanal, cekirdek):
    """
    Tek kanallı görüntüye 2D konvolüsyon uygular.
    Kenar dolgusu: sıfır (zero-padding).
    """
    h, w = kanal.shape
    ch, cw = cekirdek.shape
    pad_h, pad_w = ch // 2, cw // 2
 
    dolgulu = np.pad(kanal, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    sonuc = np.zeros_like(kanal, dtype=np.float64)
 
    for i in range(h):
        for j in range(w):
            bolge = dolgulu[i:i + ch, j:j + cw]
            sonuc[i, j] = np.sum(bolge * cekirdek)
 
    return np.clip(sonuc, 0, 255).astype(np.uint8)
 
 
def _kanal_uygula(img, cekirdek):
    """Renkli veya gri görüntüye çekirdeği uygular."""
    if len(img.shape) == 2:
        return _konvolüsyon(img, cekirdek)
    kanallar = [_konvolüsyon(img[:, :, c], cekirdek) for c in range(img.shape[2])]
    return np.stack(kanallar, axis=2)
 
 
def ortalama_blur(img, cekirdek_boyutu=3):
    """
    Kutu (ortalama) bulanıklaştırma: tüm komşu pikseller eşit ağırlıklıdır.
    Hızlıdır ama Gaussian'a göre daha az doğal görünür.
    """
    # Her elemanı eşit ağırlıklı bir çekirdek (kutupsal blur)
    cekirdek = np.ones((cekirdek_boyutu, cekirdek_boyutu), dtype=np.float64) / (cekirdek_boyutu ** 2)
    return _kanal_uygula(img, cekirdek)
 
 
def agirlikli_ortalama_blur(img, cekirdek_boyutu=3):
    """
    Ağırlıklı ortalama bulanıklaştırma: merkeze daha yakın pikseller daha fazla ağırlık taşır.
    Gaussian bulanıklığının basit bir yaklaşımıdır.
    Örnek 3x3 çekirdek: merkez=4, komşu=2, köşe=1 (toplam=16)
    """
    if cekirdek_boyutu == 3:
        cekirdek = np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=np.float64) / 16.0
    elif cekirdek_boyutu == 5:
        cekirdek = np.array([[1,  4,  6,  4, 1],
                              [4, 16, 24, 16, 4],
                              [6, 24, 36, 24, 6],
                              [4, 16, 24, 16, 4],
                              [1,  4,  6,  4, 1]], dtype=np.float64) / 256.0
    else:
        # Genel durum: kutupsal blur kullan
        return ortalama_blur(img, cekirdek_boyutu)
 
    return _kanal_uygula(img, cekirdek)
 
 
def motion_blur(img, uzunluk=9, yon='yatay'):
    """
    Hareket bulanıklığı simüle eder.
    yon: 'yatay', 'dikey' veya 'capraz'
    """
    cekirdek = np.zeros((uzunluk, uzunluk), dtype=np.float64)
 
    if yon == 'yatay':
        cekirdek[uzunluk // 2, :] = 1.0 / uzunluk
    elif yon == 'dikey':
        cekirdek[:, uzunluk // 2] = 1.0 / uzunluk
    elif yon == 'capraz':
        for i in range(uzunluk):
            cekirdek[i, i] = 1.0 / uzunluk
    else:
        raise ValueError("yon: 'yatay', 'dikey' veya 'capraz' olmalıdır.")
 
    return _kanal_uygula(img, cekirdek)