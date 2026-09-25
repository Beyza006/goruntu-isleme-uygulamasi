import numpy as np
import random
 
# ── Gürültü Ekleme ────────────────────────────────────────────────────────────
 
def salt_pepper_gurultu_ekle(img, oran=0.02):
    """
    Görüntüye tuz-biber (salt & pepper) gürültüsü ekler.
    Her piksel belirli bir olasılıkla ya tamamen siyah (0) ya da beyaz (255) yapılır.
    oran: gürültülü piksel oranı (0.02 = %2)
    """
    gurultulu = img.copy()
    h, w = img.shape[:2]
    gurultu_sayisi = int(oran * h * w)
 
    for _ in range(gurultu_sayisi):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        deger = 0 if random.random() < 0.5 else 255
 
        if img.ndim == 2:
            gurultulu[x, y] = deger
        else:
            gurultulu[x, y] = [deger] * 3  # Tüm kanallara uygula
 
    return gurultulu
 
 
# ── Mean (Ortalama) Filtre ────────────────────────────────────────────────────
 
def mean_filtre(img, cekirdek_boyutu=3):
    """
    Her pikseli komşularının ortalamasıyla değiştirir.
    Gürültüyü yumuşatır ancak kenarları biraz bulanıklaştırır.
    """
    h, w = img.shape[:2]
    pad = cekirdek_boyutu // 2
 
    if img.ndim == 2:
        kanal_sayisi = 1
        img_isle = img[:, :, np.newaxis]
    else:
        kanal_sayisi = img.shape[2]
        img_isle = img
 
    sonuc = np.zeros_like(img_isle, dtype=np.uint8)
 
    for c in range(kanal_sayisi):
        for i in range(h):
            for j in range(w):
                toplam = 0
                sayac = 0
                # Penceredeki komşu pikselleri topla
                for ki in range(-pad, pad + 1):
                    for kj in range(-pad, pad + 1):
                        ni, nj = i + ki, j + kj
                        if 0 <= ni < h and 0 <= nj < w:
                            toplam += int(img_isle[ni, nj, c])
                            sayac += 1
                sonuc[i, j, c] = toplam // sayac
 
    return sonuc[:, :, 0] if img.ndim == 2 else sonuc
 
 
# ── Median (Ortanca) Filtre ───────────────────────────────────────────────────
 
def _sirala(arr):
    """Kabarcık sıralaması (bubble sort) ile listeyi sıralar."""
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
 
def median_filtre(img, cekirdek_boyutu=3):
    """
    Her pikseli komşularının medyanıyla (ortancasıyla) değiştirir.
    Tuz-biber gürültüsüne karşı mean filtreden daha etkilidir,
    çünkü aşırı (0 veya 255) değerleri ortalamayla kirletmez.
    """
    if cekirdek_boyutu % 2 == 0 or cekirdek_boyutu < 3:
        raise ValueError("Çekirdek boyutu 3 veya üzeri tek sayı olmalıdır.")
 
    h, w = img.shape[:2]
    pad = cekirdek_boyutu // 2
 
    if img.ndim == 2:
        kanal_sayisi = 1
        img_isle = img[:, :, np.newaxis]
    else:
        kanal_sayisi = img.shape[2]
        img_isle = img
 
    # Kenarları yansıtma ile doldur
    dolgulu = np.pad(img_isle, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    sonuc = np.zeros_like(img_isle, dtype=np.uint8)
 
    for c in range(kanal_sayisi):
        for i in range(h):
            for j in range(w):
                # Penceredeki değerleri topla, sırala, ortancayı al
                komsu = dolgulu[i:i + cekirdek_boyutu, j:j + cekirdek_boyutu, c].flatten().tolist()
                siralı = _sirala(komsu)
                sonuc[i, j, c] = siralı[len(siralı) // 2]
 
    return sonuc[:, :, 0] if img.ndim == 2 else sonuc