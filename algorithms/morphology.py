import numpy as np
 
def _dolgulu_yap(img, pad, dolgu_degeri):
    """Görüntünün kenarlarını sabit değerle doldurur."""
    if img.ndim == 2:
        return np.pad(img, pad, mode='constant', constant_values=dolgu_degeri)
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=dolgu_degeri)
 
 
def genisletme(img, yapi_elemani_boyutu=3):
    """
    Genişletme (Dilation): parlak bölgeleri büyütür, karanlıkları küçültür.
    Her piksel değeri, komşuluk penceresindeki maksimum değerle değiştirilir.
    """
    se = yapi_elemani_boyutu
    pad = se // 2
    dolgulu = _dolgulu_yap(img, pad, dolgu_degeri=0)
    sonuc = np.zeros_like(img)
 
    if img.ndim == 2:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                bolge = dolgulu[i:i+se, j:j+se]
                sonuc[i, j] = np.max(bolge)
    else:
        for c in range(img.shape[2]):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    bolge = dolgulu[i:i+se, j:j+se, c]
                    sonuc[i, j, c] = np.max(bolge)
 
    return sonuc
 
 
def erozyon(img, yapi_elemani_boyutu=3):
    """
    Erozyon (Erosion): karanlık bölgeleri büyütür, parlakları küçültür.
    Her piksel değeri, komşuluk penceresindeki minimum değerle değiştirilir.
    """
    se = yapi_elemani_boyutu
    pad = se // 2
    dolgulu = _dolgulu_yap(img, pad, dolgu_degeri=255)
    sonuc = np.zeros_like(img)
 
    if img.ndim == 2:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                bolge = dolgulu[i:i+se, j:j+se]
                sonuc[i, j] = np.min(bolge)
    else:
        for c in range(img.shape[2]):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    bolge = dolgulu[i:i+se, j:j+se, c]
                    sonuc[i, j, c] = np.min(bolge)
 
    return sonuc
 
 
def acma(img, yapi_elemani_boyutu=3):
    """
    Açma (Opening): Önce erozyon sonra genişletme.
    Küçük parlak gürültü noktalarını temizler, genel yapıyı korur.
    """
    return genisletme(erozyon(img, yapi_elemani_boyutu), yapi_elemani_boyutu)
 
 
def kapama(img, yapi_elemani_boyutu=3):
    """
    Kapama (Closing): Önce genişletme sonra erozyon.
    Küçük karanlık delikleri kapatır, genel yapıyı korur.
    """
    return erozyon(genisletme(img, yapi_elemani_boyutu), yapi_elemani_boyutu)
 
 
def morfolojik_islem(img, islem, yapi_elemani_boyutu=3, tekrar=1):
    """
    islem: 'genisletme', 'erozyon', 'acma' veya 'kapama'
    tekrar: işlemi kaç kez uygulayacağı
    """
    islem_map = {
        'genisletme': genisletme,
        'erozyon': erozyon,
        'acma': acma,
        'kapama': kapama,
    }
    if islem not in islem_map:
        raise ValueError(f"Geçersiz işlem: '{islem}'. Seçenekler: {list(islem_map.keys())}")
 
    sonuc = img.copy()
    for _ in range(tekrar):
        sonuc = islem_map[islem](sonuc, yapi_elemani_boyutu)
 
    return sonuc