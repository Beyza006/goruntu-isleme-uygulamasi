import numpy as np
 
def parlaklik_ayarla(img, deger):
    """
    Her piksele sabit bir değer ekleyerek parlaklığı artırır veya azaltır.
    deger > 0 → aydınlatır, deger < 0 → karartır.
    Sonuç 0-255 aralığında tutulur.
    """
    sonuc = np.clip(img.astype(int) + deger, 0, 255)
    return sonuc.astype(np.uint8)
 
def kontrast_ayarla(img, oran):
    """
    Kontrastı ayarlar: piksel değerlerini görüntü ortalamasına göre uzaklaştırır/yaklaştırır.
    oran > 1 → kontrast artar, 0 < oran < 1 → kontrast azalır.
    """
    ortalama = np.mean(img)
    sonuc = np.clip(oran * img.astype(float) + (1 - oran) * ortalama, 0, 255)
    return sonuc.astype(np.uint8)