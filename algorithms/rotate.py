import math
import numpy as np
 
def goruntu_dondur(img, aci_derece):
    """
    Görüntüyü merkez etrafında verilen açı kadar döndürür.
    Ters haritalama: çıktı pikselinden giriş pikseline gidilir,
    bu sayede boşluk (delik) oluşmaz.
    """
    aci_radyan = math.radians(aci_derece)
    cos_a = math.cos(aci_radyan)
    sin_a = math.sin(aci_radyan)
 
    yukseklik, genislik = img.shape[:2]
    cx, cy = genislik // 2, yukseklik // 2  # Merkez nokta
 
    sonuc = np.zeros_like(img)
 
    for y in range(yukseklik):
        for x in range(genislik):
            # Çıktı pikselinin merkeze göre konumu
            x_rel = x - cx
            y_rel = y - cy
 
            # Ters dönüşüm: bu çıktı pikselinin geldiği kaynak piksel
            x_kaynak = int(cx + x_rel * cos_a + y_rel * sin_a)
            y_kaynak = int(cy - x_rel * sin_a + y_rel * cos_a)
 
            # Kaynak piksel görüntü içindeyse kopyala
            if 0 <= x_kaynak < genislik and 0 <= y_kaynak < yukseklik:
                sonuc[y, x] = img[y_kaynak, x_kaynak]
 
    return sonuc