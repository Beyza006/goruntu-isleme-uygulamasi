import numpy as np
 
def global_esikleme(img, esik=128):
    """
    Tüm görüntüye tek bir eşik değeri uygular.
    Homojen aydınlatmalı görüntülerde iyi çalışır.
    """
    gri = img if len(img.shape) == 2 else (0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]).astype(np.uint8)
    return np.where(gri > esik, 255, 0).astype(np.uint8)
 
 
def adaptif_esikleme(img, blok_boyutu=11, C=2):
    """
    Adaptif eşikleme: her piksel için komşu bloğun ortalaması eşik olarak kullanılır.
    Bu yöntem, görüntü üzerinde aydınlatma değişken olsa bile iyi sonuç verir.
 
    blok_boyutu: komşuluk penceresi (tek sayı, ör: 11)
    C: ortalamadan çıkarılacak sabit (ince ayar için)
    """
    if blok_boyutu % 2 == 0:
        raise ValueError("Blok boyutu tek sayı olmalıdır.")
 
    # Renkli ise griye çevir
    if len(img.shape) == 3:
        gri = (0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]).astype(np.uint8)
    else:
        gri = img.copy()
 
    h, w = gri.shape
    pad = blok_boyutu // 2
    sonuc = np.zeros((h, w), dtype=np.uint8)
 
    # Kenarları yansıtma yöntemiyle doldur (siyah kenar oluşmasın)
    dolgulu = np.pad(gri, pad, mode='reflect')
 
    for i in range(h):
        for j in range(w):
            # Piksel etrafındaki blokun ortalamasını hesapla
            blok = dolgulu[i:i + blok_boyutu, j:j + blok_boyutu]
            yerel_esik = blok.mean() - C
 
            # Piksel eşikten büyükse beyaz, değilse siyah
            sonuc[i, j] = 255 if gri[i, j] > yerel_esik else 0
 
    return sonuc
 
 
def cift_esikleme(img, dusuk=50, yuksek=150):
    """
    Çift eşikleme: pikselleri üç kategoriye ayırır:
    - 255 (güçlü): yüksek eşiğin üstünde
    - 128 (zayıf) : iki eşik arasında
    -   0 (yok)   : düşük eşiğin altında
    Kenar tespitinde (Sobel sonrasında) sıkça kullanılır.
    """
    if len(img.shape) == 3:
        gri = (0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]).astype(np.uint8)
    else:
        gri = img.copy()
 
    h, w = gri.shape
    sonuc = np.zeros((h, w), dtype=np.uint8)
 
    for i in range(h):
        for j in range(w):
            piksel = int(gri[i, j])
            if piksel >= yuksek:
                sonuc[i, j] = 255   # Kesinlikle kenar
            elif piksel >= dusuk:
                sonuc[i, j] = 128   # Belirsiz (zayıf kenar)
            # else: 0 (kenar değil)
 
    return sonuc