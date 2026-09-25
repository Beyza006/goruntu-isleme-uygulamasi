import numpy as np
from algorithms.gray_conversion import griye_cevir
 
def binary_donusum(img, esik=128):
    """
    Görüntüyü siyah-beyaz (binary) formata çevirir.
    Piksel > esik ise 255 (beyaz), değilse 0 (siyah).
    """
    gri = griye_cevir(img)
 
    # Eşik değerine göre her pikseli 0 ya da 255 yap
    binary = np.where(gri > esik, 255, 0).astype(np.uint8)
    return binary