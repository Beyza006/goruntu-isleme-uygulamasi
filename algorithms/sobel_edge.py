import numpy as np
import math
 
# Sobel çekirdekleri: yatay ve dikey gradyanı hesaplar
SOBEL_X = [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]
 
SOBEL_Y = [[-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]]
 
 
def _griye_cevir(img):
    """BGR görüntüyü gri tonlamaya çevirir."""
    if len(img.shape) == 2:
        return img.astype(np.float64)
    return (0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2])
 
 
def _konvolüsyon_3x3(kanal, cekirdek):
    """3x3 çekirdekle konvolüsyon uygular, sıfır doldurma kullanır."""
    h, w = kanal.shape
    sonuc = np.zeros((h, w), dtype=np.float64)
    dolgulu = np.pad(kanal, 1, mode='constant', constant_values=0)
 
    for i in range(h):
        for j in range(w):
            bolge = dolgulu[i:i+3, j:j+3]
            toplam = 0.0
            for ki in range(3):
                for kj in range(3):
                    toplam += bolge[ki, kj] * cekirdek[ki][kj]
            sonuc[i, j] = toplam
 
    return sonuc
 
 
def sobel_kenar_bul(img, esik=50):
    """
    Sobel operatörü ile kenar tespiti yapar.
    Yatay (Gx) ve dikey (Gy) gradyanlar ayrı ayrı hesaplanır.
    Kenar şiddeti: G = sqrt(Gx² + Gy²)
 
    esik: bu değerin altındaki zayıf kenarlar bastırılır (0 yapılır)
    Döndürür: kenar haritası, Gx, Gy
    """
    gri = _griye_cevir(img)
 
    # Yatay ve dikey gradyanları hesapla
    Gx = _konvolüsyon_3x3(gri, SOBEL_X)
    Gy = _konvolüsyon_3x3(gri, SOBEL_Y)
 
    # Gradyan büyüklüğü
    G = np.sqrt(Gx**2 + Gy**2)
 
    # 0-255'e normalize et
    G_norm = np.clip(G / G.max() * 255, 0, 255).astype(np.uint8) if G.max() > 0 else G.astype(np.uint8)
 
    # Eşik uygula: zayıf kenarları sil
    G_norm[G_norm < esik] = 0
 
    Gx_goster = np.clip(np.abs(Gx) / np.abs(Gx).max() * 255, 0, 255).astype(np.uint8) if Gx.max() != 0 else Gx.astype(np.uint8)
    Gy_goster = np.clip(np.abs(Gy) / np.abs(Gy).max() * 255, 0, 255).astype(np.uint8) if Gy.max() != 0 else Gy.astype(np.uint8)
 
    return G_norm, Gx_goster, Gy_goster
 
 
def gradyan_yonu(Gx, Gy):
    """
    Her pikselin gradyan yönünü (açısını) derece cinsinden döndürür.
    Kenar tespiti analizinde kullanılabilir.
    """
    return np.degrees(np.arctan2(Gy.astype(float), Gx.astype(float)))