import numpy as np
from algorithms.zoom import goruntu_olcekle
 
def _boyut_esitle(img1, img2):
    """İki görüntünün boyutu farklıysa img2'yi img1'in boyutuna uydur."""
    h, w = img1.shape[:2]

    if img2.shape[:2] != (h, w):
        olcek_h = h / img2.shape[0]
        olcek_w = w / img2.shape[1]

        # En az bir eksen küçük kalmasın diye büyük olan oran seçilir
        olcek = max(olcek_h, olcek_w)
        img2 = goruntu_olcekle(img2, olcek, yontem='nearest')

        yeni_h, yeni_w = img2.shape[:2]

        # Eğer büyük geldiyse kırp
        if yeni_h > h:
            bas_y = (yeni_h - h) // 2
            img2 = img2[bas_y:bas_y + h, :]
        if yeni_w > w:
            bas_x = (yeni_w - w) // 2
            img2 = img2[:, bas_x:bas_x + w]

        # Eğer hala küçük kaldıysa üstten/sağdan doldur
        if img2.shape[0] < h or img2.shape[1] < w:
            if len(img2.shape) == 3:
                sonuc = np.zeros((h, w, img2.shape[2]), dtype=img2.dtype)
            else:
                sonuc = np.zeros((h, w), dtype=img2.dtype)

            sonuc[:img2.shape[0], :img2.shape[1]] = img2
            img2 = sonuc

    return img1, img2

 
def goruntu_topla(img1, img2):
    """
    İki görüntüyü piksel piksel toplar.
    Sonuç 0-255 aralığında tutulur (taşma olursa kesilir).
    """
    img1, img2 = _boyut_esitle(img1, img2)
    sonuc = np.clip(img1.astype(int) + img2.astype(int), 0, 255)
    return sonuc.astype(np.uint8)
 
def goruntu_cikar(img1, img2):
    """
    img2'yi img1'den piksel piksel çıkarır.
    Negatif değerler 0'a sabitlenir.
    """
    img1, img2 = _boyut_esitle(img1, img2)
    sonuc = np.clip(img1.astype(int) - img2.astype(int), 0, 255)
    return sonuc.astype(np.uint8)
 
def goruntu_carp(img1, img2):
    """
    İki görüntüyü piksel piksel çarpar.
    Her iki görüntü de 0-1'e normalize edilerek çarpılır, ardından 0-255'e geri ölçeklenir.
    """
    img1, img2 = _boyut_esitle(img1, img2)
    sonuc = np.clip((img1.astype(float) * img2.astype(float)) / 255.0, 0, 255)
    return sonuc.astype(np.uint8)
 
def aritmetik_islem(img1, img2, islem):
    """
    islem: 'toplama', 'cikarma' veya 'carpma'
    """
    if islem == 'toplama':
        return goruntu_topla(img1, img2)
    elif islem == 'cikarma':
        return goruntu_cikar(img1, img2)
    elif islem == 'carpma':
        return goruntu_carp(img1, img2)
    else:
        raise ValueError(f"Geçersiz işlem: '{islem}'. 'toplama', 'cikarma' veya 'carpma' olmalı.")