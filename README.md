# Görüntü İşleme Uygulaması 🖼️

Bu proje, Görüntü İşleme dersi kapsamında 4 kişilik grubumuz tarafından geliştirilmiştir. Temel görüntü işleme algoritmalarını sıfırdan implement ederek interaktif bir masaüstü arayüzüne entegre ettik.

## 🎯 Projenin Amacı
OpenCV ve NumPy gibi kütüphaneler aracılığıyla hazır fonksiyon çağırmak yerine, algoritmalar sıfırdan Python ile yazılmıştır. Her modül bağımsız test edilebilir ve genişletilebilir şekilde tasarlanmıştır.

## 🛠️ Kullanılan Teknolojiler
- **Dil:** Python
- **Arayüz:** Tkinter (CustomTkinter stili)
- **Görüntü İşleme:** OpenCV, NumPy, Pillow
- **Görselleştirme:** Matplotlib

## 🔧 Desteklenen Algoritmalar (15 Modül)

| # | Algoritma | Açıklama |
|---|-----------|----------|
| 1 | Gri Dönüşüm | BGR → Gri tonlama |
| 2 | Binary Dönüşüm | Eşik değerine göre siyah-beyaz |
| 3 | Döndürme | Açı bazlı görüntü döndürme |
| 4 | Kırpma | Koordinat bazlı bölge kırpma |
| 5 | Zoom | Nearest Neighbor / Bilinear interpolasyon |
| 6 | Renk Uzayı Dönüşümü | RGB, HSV, LAB, YCrCb, XYZ, HLS, YUV, LUV |
| 7 | Histogram | Eşitleme, germe ve görselleştirme |
| 8 | Aritmetik İşlemler | Toplama, çıkarma, çarpma |
| 9 | Parlaklık & Kontrast | Kaydırıcı tabanlı ayar |
| 10 | Gaussian Konvolüsyon | Çekirdek boyutu ve sigma ayarlanabilir |
| 11 | Eşikleme | Global, Adaptif ve Çift Eşik |
| 12 | Sobel Kenar Bulma | Gx, Gy gradyanları ve kenar haritası |
| 13 | Gürültü & Filtreleme | Salt & Pepper, Mean Filtre, Median Filtre |
| 14 | Blurring | Ortalama, Ağırlıklı, Motion Blur |
| 15 | Morfolojik İşlemler | Genişletme, Erozyon, Açma, Kapama |

## 🚀 Nasıl Çalıştırılır?

**1. Depoyu İndirin:**
```bash
git clone https://github.com/Beyza006/image-processing-app.git
cd image-processing-app
```

**2. Gerekli Kütüphaneleri Kurun:**
```bash
pip install -r requirements.txt
```

**3. Uygulamayı Başlatın:**
```bash
python app.py
```
> **Not:** Uygulama açıldığında `images/` klasöründeki ilk görsel otomatik olarak yüklenir. İsterseniz arayüz üzerinden farklı bir görsel de seçebilirsiniz.

## 📁 Proje Yapısı
```
Image_Processing_Project/
│
├── app.py                  # Ana masaüstü arayüzü (GUI)
├── requirements.txt        # Gerekli kütüphaneler
├── images/                 # Örnek test görselleri
│
└── algorithms/             # Tüm algoritmalar bağımsız modüller halinde
    ├── sobel_edge.py
    ├── histogram.py
    ├── morphology.py
    ├── blurring.py
    ├── gaussian.py
    ├── zoom.py
    ├── rotate.py
    ├── crop.py
    ├── brightness.py
    ├── color_space.py
    ├── noise_and_filter.py
    ├── adaptive_threshold.py
    ├── arithmetic_ops.py
    ├── gray_conversion.py
    └── binary_conversion.py
```

---
*Bu proje, bilgisayar mühendisliği lisans eğitimi kapsamında geliştirilmiş bir akademik grup çalışmasıdır.*
