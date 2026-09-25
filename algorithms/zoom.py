import numpy as np
 
def _kanal_olcekle(kanal, olcek, yontem):
    """Tek kanallı (2D) bir diziyi ölçekler."""
    h, w = kanal.shape
    yeni_h, yeni_w = int(h * olcek), int(w * olcek)
 
    if yontem == 'nearest':
        # En yakın komşu: her çıktı pikselini en yakın kaynak piksele eşle
        y_idx = np.clip((np.arange(yeni_h) / olcek).astype(int), 0, h - 1)
        x_idx = np.clip((np.arange(yeni_w) / olcek).astype(int), 0, w - 1)
        return kanal[y_idx[:, None], x_idx]
 
    elif yontem == 'bilinear':
        # Bilinear: 4 komşu pikselin ağırlıklı ortalaması alınır
        y, x = np.meshgrid(np.arange(yeni_h), np.arange(yeni_w), indexing='ij')
        y0 = np.floor(y / olcek).astype(int)
        x0 = np.floor(x / olcek).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
 
        dy = (y / olcek) - np.floor(y / olcek)
        dx = (x / olcek) - np.floor(x / olcek)
 
        # 4 köşe pikselinin değerleri
        f00 = kanal[y0, x0]
        f01 = kanal[y0, x1]
        f10 = kanal[y1, x0]
        f11 = kanal[y1, x1]
 
        # Bilinear interpolasyon formülü
        sonuc = (1 - dy) * (1 - dx) * f00 + \
                (1 - dy) * dx       * f01 + \
                dy       * (1 - dx) * f10 + \
                dy       * dx       * f11
 
        return np.clip(sonuc, 0, 255).astype(np.uint8)
 
    else:
        raise ValueError(f"Desteklenmeyen yöntem: '{yontem}'. 'nearest' veya 'bilinear' kullanın.")
 
 
def goruntu_olcekle(img, olcek, yontem='nearest'):
    """
    Görüntüyü verilen ölçek katsayısıyla büyütür veya küçültür.
    olcek > 1 → yakınlaştırma, olcek < 1 → uzaklaştırma
    yontem: 'nearest' (hızlı) veya 'bilinear' (daha pürüzsüz)
    """
    img = np.array(img)
 
    if len(img.shape) == 3:
        # Renkli görüntü: her kanalı ayrı ölçekle
        kanallar = [_kanal_olcekle(img[:, :, c], olcek, yontem) for c in range(img.shape[2])]
        return np.stack(kanallar, axis=2)
    else:
        # Gri görüntü
        return _kanal_olcekle(img, olcek, yontem)