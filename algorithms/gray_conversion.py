import numpy as np
 
def griye_cevir(img):
    """
    RGB görüntüyü gri tonlamaya çevirir.
    Formül: Gray = 0.299*R + 0.587*G + 0.114*B
    (İnsan gözünün yeşile daha duyarlı olmasından gelir)
    """
    img_array = np.array(img)
 
    # Zaten gri ise direkt döndür
    if len(img_array.shape) == 2:
        return img_array
 
    # Her piksel için ağırlıklı ortalama hesapla (R, G, B sırası OpenCV'de BGR'dir)
    gri = np.dot(img_array[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    return gri