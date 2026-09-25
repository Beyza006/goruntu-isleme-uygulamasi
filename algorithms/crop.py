import numpy as np
 
def goruntu_kirp(img, x, y, genislik, yukseklik):
    """
    Görüntüden dikdörtgen bir bölge keser.
    x, y: sol üst köşe koordinatları
    genislik, yukseklik: kesilecek bölgenin boyutları
    """
    img_yukseklik, img_genislik = img.shape[:2]
 
    # Koordinatların görüntü sınırları içinde olup olmadığını kontrol et
    if not (0 <= x < img_genislik and 0 <= y < img_yukseklik):
        raise ValueError("Başlangıç koordinatları görüntü dışında.")
    if x + genislik > img_genislik or y + yukseklik > img_yukseklik:
        raise ValueError("Kırpma alanı görüntü sınırlarını aşıyor.")
 
    # NumPy dilimleme ile bölgeyi al
    return img[y:y + yukseklik, x:x + genislik]