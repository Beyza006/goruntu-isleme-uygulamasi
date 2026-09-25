from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.adaptive_threshold import (
    adaptif_esikleme,
    cift_esikleme,
    global_esikleme,
)
from algorithms.arithmetic_ops import aritmetik_islem
from algorithms.binary_conversion import binary_donusum
from algorithms.blurring import agirlikli_ortalama_blur, motion_blur, ortalama_blur
from algorithms.brightness import kontrast_ayarla, parlaklik_ayarla
from algorithms.color_space import (
    bgr_to_hls,
    bgr_to_hsv,
    bgr_to_lab,
    bgr_to_luv,
    bgr_to_rgb,
    bgr_to_xyz,
    bgr_to_ycrcb,
    bgr_to_yuv,
)
from algorithms.crop import goruntu_kirp
from algorithms.gaussian import gaussian_blur
from algorithms.gray_conversion import griye_cevir
from algorithms.histogram import (
    histogram_esitle_gri,
    histogram_gerdirme_renkli,
    histogram_goster,
)
from algorithms.morphology import morfolojik_islem
from algorithms.noise_and_filter import (
    mean_filtre,
    median_filtre,
    salt_pepper_gurultu_ekle,
)
from algorithms.rotate import goruntu_dondur
from algorithms.sobel_edge import sobel_kenar_bul
from algorithms.zoom import goruntu_olcekle


DESTEKLENEN_UZANTILAR = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

RENKLER = {
    "arka": "#FFFFFF",
    "yüzey": "#FFFFFF",
    "kart": "#F7F9FC",
    "kart_alt": "#EEF3FA",
    "vurgu": "#163B73",
    "vurgu_orta": "#2457A6",
    "vurgu_açık": "#D9E6F7",
    "kenar": "#D8E2F0",
    "metin": "#11284B",
    "metin_soluk": "#5E7395",
    "boşluk": "#F1F5FB",
    "durum": "#EEF3FA",
    "giriş": "#FFFFFF",
}

YAZI = {
    "başlık": ("Segoe UI Semibold", 20),
    "panel": ("Segoe UI Semibold", 13),
    "kart": ("Segoe UI Semibold", 11),
    "alt": ("Segoe UI Semibold", 10),
    "normal": ("Segoe UI", 10),
    "küçük": ("Segoe UI", 9),
    "buton": ("Segoe UI Semibold", 10),
}

YÖNTEMLER = [
    "1) Gri Dönüşüm",
    "2) Binary Dönüşüm",
    "3) Döndürme",
    "4) Kırpma",
    "5) Zoom",
    "6) Renk Uzayı Dönüşümü",
    "7) Histogram",
    "8) Aritmetik İşlemler",
    "9) Parlaklık ve Kontrast",
    "10) Gaussian Konvolüsyon",
    "11) Eşikleme İşlemleri",
    "12) Sobel Kenar Bulma",
    "13) Gürültü Ekleme ve Filtreleme",
    "14) Blurring",
    "15) Morfolojik İşlemler",
]


class ModernKaydırıcı(tk.Frame):
    def __init__(self, ebeveyn, metin, başlangıç, alt, üst, adım=1):
        super().__init__(ebeveyn, bg=RENKLER["kart"])
        self.var = tk.DoubleVar(value=başlangıç)
        self.adım = adım

        üst_satır = tk.Frame(self, bg=RENKLER["kart"])
        üst_satır.pack(fill="x")

        tk.Label(
            üst_satır,
            text=metin,
            font=YAZI["küçük"],
            bg=RENKLER["kart"],
            fg=RENKLER["metin"],
        ).pack(side="left")

        self.değer_etiketi = tk.Label(
            üst_satır,
            font=YAZI["küçük"],
            bg=RENKLER["vurgu_açık"],
            fg=RENKLER["vurgu"],
            padx=8,
            pady=3,
        )
        self.değer_etiketi.pack(side="right")

        ttk.Scale(
            self,
            from_=alt,
            to=üst,
            variable=self.var,
            orient="horizontal",
            style="Modern.Horizontal.TScale",
        ).pack(fill="x", pady=(8, 0))

        self.var.trace_add("write", self._güncelle)
        self._güncelle()

    def _güncelle(self, *_args):
        değer = self.var.get()
        if float(self.adım).is_integer():
            metin = f"{int(round(değer))}"
        else:
            metin = f"{değer:.2f}"
        self.değer_etiketi.config(text=metin)

    def get(self):
        return self.var.get()


class GörüntüİşlemeApp:
    def __init__(self, kök):
        self.kök = kök
        self.kök.title("Görüntü İşleme Uygulaması")
        self.kök.configure(bg=RENKLER["arka"])
        self.kök.minsize(1280, 760)
        try:
            self.kök.state("zoomed")
        except tk.TclError:
            self.kök.geometry("1400x860")

        self.orijinal_img = None
        self.işlenmiş_img = None
        self.ikinci_img = None
        self.gürültülü_img = None
        self.fotolar = {}
        self._yeniden_çiz = None

        self._stilleri_hazırla()
        self._arayüzü_kur()
        self._başlangıç_görselini_yükle()

    def _stilleri_hazırla(self):
        stil = ttk.Style()
        stil.theme_use("clam")
        stil.configure(
            "Modern.TCombobox",
            fieldbackground=RENKLER["giriş"],
            background=RENKLER["yüzey"],
            foreground=RENKLER["metin"],
            bordercolor=RENKLER["kenar"],
            lightcolor=RENKLER["kenar"],
            darkcolor=RENKLER["kenar"],
            arrowcolor=RENKLER["vurgu"],
            padding=8,
        )
        stil.configure(
            "Modern.Vertical.TScrollbar",
            background=RENKLER["vurgu_açık"],
            troughcolor=RENKLER["kart"],
            bordercolor=RENKLER["kart"],
            arrowcolor=RENKLER["vurgu"],
        )
        stil.map(
            "Modern.Vertical.TScrollbar",
            background=[("active", RENKLER["vurgu_orta"])],
        )
        stil.configure(
            "Modern.Horizontal.TScale",
            background=RENKLER["kart"],
            troughcolor=RENKLER["vurgu_açık"],
            sliderthickness=18,
        )

    def _arayüzü_kur(self):
        self._üst_başlık()

        gövde = tk.Frame(self.kök, bg=RENKLER["arka"])
        gövde.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self._sol_panel(gövde)
        self._sağ_panel(gövde)
        self._durum_çubuğu()

        self.kök.bind("<Configure>", self._boyut_değişti)

    def _üst_başlık(self):
        çerçeve = tk.Frame(self.kök, bg=RENKLER["arka"], padx=20, pady=20)
        çerçeve.pack(fill="x")

        başlık_kartı = tk.Frame(
            çerçeve,
            bg=RENKLER["yüzey"],
            highlightbackground=RENKLER["kenar"],
            highlightthickness=1,
            padx=18,
            pady=16,
        )
        başlık_kartı.pack(fill="x")

        tk.Label(
            başlık_kartı,
            text="Görüntü İşleme Kontrol Paneli",
            font=YAZI["başlık"],
            bg=RENKLER["yüzey"],
            fg=RENKLER["metin"],
        ).pack(anchor="w")

    def _sol_panel(self, ebeveyn):
        panel = tk.Frame(
            ebeveyn,
            bg=RENKLER["yüzey"],
            width=390,
            highlightbackground=RENKLER["kenar"],
            highlightthickness=1,
        )
        panel.pack(side="left", fill="y", padx=(0, 18))
        panel.pack_propagate(False)

        canvas = tk.Canvas(
            panel,
            bg=RENKLER["yüzey"],
            highlightthickness=0,
            bd=0,
            width=388,
        )
        kaydırma = ttk.Scrollbar(
            panel,
            orient="vertical",
            command=canvas.yview,
            style="Modern.Vertical.TScrollbar",
        )
        canvas.configure(yscrollcommand=kaydırma.set)

        self.panel_içerik = tk.Frame(canvas, bg=RENKLER["yüzey"])
        self.panel_içerik.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.panel_içerik, anchor="nw", width=366)
        canvas.pack(side="left", fill="both", expand=True)
        kaydırma.pack(side="right", fill="y")

        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"),
        )

        self._dosya_kartı()
        self._yöntem_seçici()

        self.yöntem_içerik = tk.Frame(self.panel_içerik, bg=RENKLER["yüzey"])
        self.yöntem_içerik.pack(fill="x", padx=10, pady=(0, 10))
        self._yöntem_panelini_göster(self.yöntem_var.get())

    def _sağ_panel(self, ebeveyn):
        alan = tk.Frame(ebeveyn, bg=RENKLER["arka"])
        alan.pack(side="left", fill="both", expand=True)

        üst = tk.Frame(
            alan,
            bg=RENKLER["yüzey"],
            highlightbackground=RENKLER["kenar"],
            highlightthickness=1,
            padx=18,
            pady=14,
        )
        üst.pack(fill="x", pady=(0, 14))

        tk.Frame(üst, bg=RENKLER["yüzey"], height=4).pack(anchor="w")

        self.önizleme = tk.Frame(alan, bg=RENKLER["arka"])
        self.önizleme.pack(fill="both", expand=True)
        self.önizleme.grid_columnconfigure(0, weight=1)
        self.önizleme.grid_columnconfigure(1, weight=1)
        self.önizleme.grid_columnconfigure(2, weight=1)
        self.önizleme.grid_rowconfigure(0, weight=1)

        self.orijinal_kart, self.label_orijinal = self._önizleme_kartı(
            self.önizleme,
            "Orijinal Görüntü",
            "Başlangıç görseli yükleniyor...",
        )
        self.gürültü_kart, self.label_gürültülü = self._önizleme_kartı(
            self.önizleme,
            "Gürültülü Görüntü",
            "Gürültü eklendiğinde burada görünecek.",
        )
        self.işlenmiş_kart, self.label_işlenmiş = self._önizleme_kartı(
            self.önizleme,
            "İşlenmiş Görüntü",
            "İşlem sonucu burada görünecek.",
        )

        self._önizleme_modu("iki")

    def _durum_çubuğu(self):
        self.durum_var = tk.StringVar(
            value="Hazır. Images klasöründeki ilk görsel otomatik olarak yüklenecek."
        )
        tk.Label(
            self.kök,
            textvariable=self.durum_var,
            font=YAZI["küçük"],
            bg=RENKLER["durum"],
            fg=RENKLER["metin"],
            anchor="w",
            padx=20,
            pady=10,
        ).pack(fill="x", side="bottom")

    def _kart(self, başlık, açıklama=None, ebeveyn=None):
        ebeveyn = ebeveyn or self.panel_içerik
        kart = tk.Frame(
            ebeveyn,
            bg=RENKLER["kart"],
            highlightbackground=RENKLER["kenar"],
            highlightthickness=1,
            padx=12,
            pady=12,
        )
        kart.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(
            kart,
            text=başlık,
            font=YAZI["kart"],
            bg=RENKLER["kart"],
            fg=RENKLER["metin"],
        ).pack(anchor="w")

        if açıklama:
            tk.Label(
                kart,
                text=açıklama,
                font=YAZI["küçük"],
                bg=RENKLER["kart"],
                fg=RENKLER["metin_soluk"],
                wraplength=310,
                justify="left",
            ).pack(anchor="w", pady=(4, 10))

        return kart

    def _alt_başlık(self, ebeveyn, metin):
        tk.Label(
            ebeveyn,
            text=metin,
            font=YAZI["alt"],
            bg=RENKLER["kart"],
            fg=RENKLER["vurgu"],
        ).pack(anchor="w", pady=(8, 4))

    def _buton(self, ebeveyn, metin, komut):
        tk.Button(
            ebeveyn,
            text=metin,
            command=komut,
            font=YAZI["buton"],
            bg=RENKLER["vurgu"],
            fg="white",
            activebackground=RENKLER["vurgu_orta"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=9,
            cursor="hand2",
        ).pack(fill="x", pady=(8, 0))

    def _combobox(self, ebeveyn, seçenekler, başlangıç=0):
        var = tk.StringVar(value=seçenekler[başlangıç])
        ttk.Combobox(
            ebeveyn,
            textvariable=var,
            values=seçenekler,
            state="readonly",
            style="Modern.TCombobox",
            font=YAZI["normal"],
        ).pack(fill="x")
        return var

    def _spinbox(self, ebeveyn, etiket, değer):
        satır = tk.Frame(ebeveyn, bg=RENKLER["kart"])
        satır.pack(fill="x", pady=(0, 4))

        tk.Label(
            satır,
            text=etiket,
            font=YAZI["küçük"],
            bg=RENKLER["kart"],
            fg=RENKLER["metin"],
            width=11,
            anchor="w",
        ).pack(side="left")

        var = tk.IntVar(value=değer)
        tk.Spinbox(
            satır,
            from_=0,
            to=9999,
            textvariable=var,
            width=10,
            font=YAZI["küçük"],
            bg=RENKLER["giriş"],
            fg=RENKLER["metin"],
            buttonbackground=RENKLER["kart_alt"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=RENKLER["kenar"],
        ).pack(side="right")
        return var

    def _kaydırıcı(self, ebeveyn, metin, başlangıç, alt, üst, adım=1):
        bileşen = ModernKaydırıcı(ebeveyn, metin, başlangıç, alt, üst, adım)
        bileşen.pack(fill="x", pady=(2, 0))
        return bileşen

    def _önizleme_kartı(self, ebeveyn, başlık, metin):
        kart = tk.Frame(
            ebeveyn,
            bg=RENKLER["yüzey"],
            highlightbackground=RENKLER["kenar"],
            highlightthickness=1,
            padx=12,
            pady=12,
        )

        tk.Label(
            kart,
            text=başlık,
            font=YAZI["panel"],
            bg=RENKLER["yüzey"],
            fg=RENKLER["metin"],
        ).pack(anchor="w", pady=(0, 10))

        etiket = tk.Label(
            kart,
            text=metin,
            bg=RENKLER["boşluk"],
            fg=RENKLER["metin_soluk"],
            font=YAZI["normal"],
            justify="center",
            wraplength=360,
            highlightbackground=RENKLER["kenar"],
            highlightthickness=1,
        )
        etiket.pack(fill="both", expand=True)
        etiket.bind("<Configure>", self._boyut_değişti)
        return kart, etiket

    def _dosya_kartı(self):
        kart = self._kart(
            "Dosya İşlemleri",
            "Uygulama açıldığında images klasöründeki ilk görsel otomatik yüklenir.",
        )
        self._buton(kart, "Başka Görsel Aç", self._görüntü_aç)
        self._buton(kart, "İkinci Görsel Aç", self._ikinci_görüntü_aç)
        self._buton(kart, "Sonucu Kaydet", self._görüntü_kaydet)

    def _yöntem_seçici(self):
        kart = self._kart(
            "İşlem Seçimi",
            "Aşağıdan bir yöntem seç. Sol panelde yalnızca seçtiğin yöntemin ayarları gösterilir.",
        )
        self.yöntem_var = tk.StringVar(value=YÖNTEMLER[0])
        kutu = ttk.Combobox(
            kart,
            textvariable=self.yöntem_var,
            values=YÖNTEMLER,
            state="readonly",
            style="Modern.TCombobox",
            font=YAZI["normal"],
        )
        kutu.pack(fill="x")
        kutu.bind("<<ComboboxSelected>>", lambda _e: self._yöntem_panelini_göster(self.yöntem_var.get()))

    def _yöntem_panelini_temizle(self):
        for bileşen in self.yöntem_içerik.winfo_children():
            bileşen.destroy()

    def _yöntem_panelini_göster(self, yöntem):
        self._yöntem_panelini_temizle()
        kart = self._kart(yöntem, ebeveyn=self.yöntem_içerik)

        if yöntem == "1) Gri Dönüşüm":
            self._buton(kart, "Griye Çevir", self._gri_uygula)

        elif yöntem == "2) Binary Dönüşüm":
            self.binary_eşik = self._kaydırıcı(kart, "Eşik değeri", 128, 0, 255)
            self._buton(kart, "Binary Uygula", self._binary_uygula)

        elif yöntem == "3) Döndürme":
            self.dönüş_açı = self._kaydırıcı(kart, "Açı", 45, -180, 180)
            self._buton(kart, "Döndür", self._döndür_uygula)

        elif yöntem == "4) Kırpma":
            self.kırp_x = self._spinbox(kart, "X", 50)
            self.kırp_y = self._spinbox(kart, "Y", 50)
            self.kırp_w = self._spinbox(kart, "Genişlik", 200)
            self.kırp_h = self._spinbox(kart, "Yükseklik", 200)
            self._buton(kart, "Kırp", self._kırp_uygula)

        elif yöntem == "5) Zoom":
            self.ölçek = self._kaydırıcı(kart, "Ölçek", 1.5, 0.1, 4.0, 0.1)
            self.zoom_yöntem = self._combobox(kart, ["nearest", "bilinear"])
            self._buton(kart, "Ölçekle", self._zoom_uygula)

        elif yöntem == "6) Renk Uzayı Dönüşümü":
            self.renk_uzayı = self._combobox(
                kart,
                ["RGB", "HSV", "YCrCb", "LAB", "XYZ", "HLS", "YUV", "LUV"],
            )
            self._buton(kart, "Dönüştür", self._renk_uzayı_uygula)

        elif yöntem == "7) Histogram":
            self._alt_başlık(kart, "Düzeltme")
            self._buton(kart, "Histogram Eşitleme", self._histogram_eşitle)
            self._buton(kart, "Histogram Germe", self._histogram_ger)
            self._alt_başlık(kart, "Görselleştirme")
            self._buton(kart, "Histogram Göster", self._histogram_göster)

        elif yöntem == "8) Aritmetik İşlemler":
            self.aritmetik_işlem = self._combobox(kart, ["toplama", "cikarma", "carpma"])
            self._buton(kart, "İşlemi Uygula", self._aritmetik_uygula)

        elif yöntem == "9) Parlaklık ve Kontrast":
            self.parlaklık = self._kaydırıcı(kart, "Parlaklık", 50, -150, 150)
            self._buton(kart, "Parlaklık Uygula", self._parlaklık_uygula)
            self.kontrast = self._kaydırıcı(kart, "Kontrast", 1.5, 0.1, 3.0, 0.1)
            self._buton(kart, "Kontrast Uygula", self._kontrast_uygula)

        elif yöntem == "10) Gaussian Konvolüsyon":
            self.gaussian_boyut = self._kaydırıcı(kart, "Çekirdek", 5, 3, 15)
            self.gaussian_sigma = self._kaydırıcı(kart, "Sigma", 1.0, 0.1, 5.0, 0.1)
            self._buton(kart, "Gaussian Uygula", self._gaussian_uygula)

        elif yöntem == "11) Eşikleme İşlemleri":
            self._alt_başlık(kart, "Global")
            self.eşik_global = self._kaydırıcı(kart, "Global eşik", 128, 0, 255)
            self._buton(kart, "Global Eşikleme", self._global_eşik_uygula)
            self._alt_başlık(kart, "Adaptif")
            self.adaptif_blok = self._kaydırıcı(kart, "Blok boyutu", 11, 3, 51, 2)
            self.adaptif_c = self._kaydırıcı(kart, "C sabiti", 2, 0, 20)
            self._buton(kart, "Adaptif Eşikleme", self._adaptif_eşik_uygula)
            self._alt_başlık(kart, "Çift Eşik")
            self.çift_düşük = self._kaydırıcı(kart, "Düşük eşik", 50, 0, 255)
            self.çift_yüksek = self._kaydırıcı(kart, "Yüksek eşik", 150, 0, 255)
            self._buton(kart, "Çift Eşikleme", self._çift_eşik_uygula)

        elif yöntem == "12) Sobel Kenar Bulma":
            self.sobel_eşik = self._kaydırıcı(kart, "Eşik", 50, 0, 255)
            self._buton(kart, "Sobel Uygula", self._sobel_uygula)

        elif yöntem == "13) Gürültü Ekleme ve Filtreleme":
            self.gürültü_oran = self._kaydırıcı(kart, "Gürültü oranı", 0.05, 0.01, 0.20, 0.01)
            self._buton(kart, "Salt & Pepper Ekle", self._gürültü_ekle)
            self._buton(kart, "Mean Filtre Uygula", self._mean_uygula)
            self._buton(kart, "Median Filtre Uygula", self._median_uygula)

        elif yöntem == "14) Blurring":
            self.blur_boyut = self._kaydırıcı(kart, "Çekirdek", 5, 3, 15)
            self.blur_tür = self._combobox(
                kart,
                ["ortalama", "ağırlıklı", "motion_yatay", "motion_dikey", "motion_çapraz"],
            )
            self._buton(kart, "Blur Uygula", self._blur_uygula)

        elif yöntem == "15) Morfolojik İşlemler":
            self.morfo_işlem = self._combobox(kart, ["genisletme", "erozyon", "acma", "kapama"])
            self.morfo_boyut = self._kaydırıcı(kart, "Yapı elemanı boyutu", 3, 3, 15)
            self.morfo_tekrar = self._kaydırıcı(kart, "Tekrar", 1, 1, 5)
            self._buton(kart, "Morfoloji Uygula", self._morfoloji_uygula)

    def _başlangıç_görselini_yükle(self):
        klasör = Path("images")
        if not klasör.exists():
            self.durum_var.set("Images klasörü bulunamadı. Başka görsel açarak devam edebilirsin.")
            return

        dosyalar = sorted(
            dosya for dosya in klasör.iterdir()
            if dosya.is_file() and dosya.suffix.lower() in DESTEKLENEN_UZANTILAR
        )
        if not dosyalar:
            self.durum_var.set("Images klasöründe desteklenen görsel bulunamadı.")
            return

        self._görsel_yükle(dosyalar[0], otomatik=True)

    def _görüntü_oku(self, yol):
        veri = np.fromfile(str(yol), dtype=np.uint8)
        if veri.size == 0:
            return None
        return cv2.imdecode(veri, cv2.IMREAD_COLOR)

    def _görüntü_yaz(self, yol, img):
        uzantı = Path(yol).suffix.lower() or ".png"
        if uzantı == ".jpg":
            uzantı = ".jpeg"
        başarılı, tampon = cv2.imencode(uzantı, img)
        if not başarılı:
            return False
        tampon.tofile(yol)
        return True

    def _görsel_yükle(self, yol, otomatik=False):
        img = self._görüntü_oku(yol)
        if img is None:
            messagebox.showerror("Hata", "Görüntü yüklenemedi.")
            return

        self.orijinal_img = img
        self.işlenmiş_img = None
        self.gürültülü_img = None
        self._önizleme_modu("iki")
        self._etiket_sıfırla(self.label_işlenmiş, "İşlem sonucu burada görünecek.")
        self._etiket_sıfırla(self.label_gürültülü, "Gürültü eklendiğinde burada görünecek.")
        self._tüm_önizlemeleri_çiz()

        ad = Path(yol).name
        boyut = f"{img.shape[1]}x{img.shape[0]} piksel"
        if otomatik:
            self.durum_var.set(f"Başlangıç görseli otomatik yüklendi: {ad} | {boyut}")
        else:
            self.durum_var.set(f"Görsel yüklendi: {ad} | {boyut}")

    def _etiket_sıfırla(self, etiket, metin):
        etiket.configure(text=metin, image="")

    def _önizleme_modu(self, mod):
        self.orijinal_kart.grid_forget()
        self.gürültü_kart.grid_forget()
        self.işlenmiş_kart.grid_forget()

        if mod == "üç":
            self.orijinal_kart.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            self.gürültü_kart.grid(row=0, column=1, sticky="nsew", padx=8)
            self.işlenmiş_kart.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        else:
            self.orijinal_kart.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            self.işlenmiş_kart.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    def _boyut_değişti(self, _event=None):
        if self._yeniden_çiz is not None:
            self.kök.after_cancel(self._yeniden_çiz)
        self._yeniden_çiz = self.kök.after(120, self._tüm_önizlemeleri_çiz)

    def _tüm_önizlemeleri_çiz(self):
        self._yeniden_çiz = None
        if self.orijinal_img is not None:
            self._görüntü_göster(self.orijinal_img, self.label_orijinal, "orijinal")
        if self.gürültülü_img is not None:
            self._görüntü_göster(self.gürültülü_img, self.label_gürültülü, "gürültülü")
        if self.işlenmiş_img is not None:
            self._görüntü_göster(self.işlenmiş_img, self.label_işlenmiş, "işlenmiş")

    def _görüntü_göster(self, img, etiket, anahtar):
        if img is None:
            return

        if len(img.shape) == 2:
            pil = Image.fromarray(img)
        else:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        etiket.update_idletasks()
        genişlik = max(etiket.winfo_width() - 10, 180)
        yükseklik = max(etiket.winfo_height() - 10, 180)
        pil.thumbnail((genişlik, yükseklik), Image.LANCZOS)

        foto = ImageTk.PhotoImage(pil)
        etiket.configure(image=foto, text="")
        self.fotolar[anahtar] = foto

    def _iki_panel_sonuç(self, img, durum):
        self.gürültülü_img = None
        self.işlenmiş_img = img
        self._önizleme_modu("iki")
        self._görüntü_göster(img, self.label_işlenmiş, "işlenmiş")
        self.durum_var.set(durum)

    def _üç_panel_sonuç(self, gürültülü, düzeltilmiş=None, durum=""):
        self.gürültülü_img = gürültülü
        self.işlenmiş_img = düzeltilmiş
        self._önizleme_modu("üç")
        self._görüntü_göster(gürültülü, self.label_gürültülü, "gürültülü")
        if düzeltilmiş is None:
            self._etiket_sıfırla(self.label_işlenmiş, "Filtre uygulandığında düzeltilmiş sonuç burada görünecek.")
        else:
            self._görüntü_göster(düzeltilmiş, self.label_işlenmiş, "işlenmiş")
        self.durum_var.set(durum)

    def _görüntü_kontrol(self):
        if self.orijinal_img is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin.")
            return False
        return True

    def _orijinal_kopya(self):
        return self.orijinal_img.copy()

    def _gürültülü_kaynak(self):
        if self.gürültülü_img is None:
            oran = float(self.gürültü_oran.get())
            self.gürültülü_img = salt_pepper_gurultu_ekle(self.orijinal_img.copy(), oran)
        return self.gürültülü_img.copy()

    def _görüntü_aç(self):
        yol = filedialog.askopenfilename(
            filetypes=[("Görüntü dosyaları", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        if yol:
            self._görsel_yükle(Path(yol))

    def _ikinci_görüntü_aç(self):
        yol = filedialog.askopenfilename(
            filetypes=[("Görüntü dosyaları", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        if not yol:
            return
        img = self._görüntü_oku(yol)
        if img is None:
            messagebox.showerror("Hata", "İkinci görüntü yüklenemedi.")
            return
        self.ikinci_img = img
        self.durum_var.set(f"İkinci görüntü yüklendi: {Path(yol).name}")

    def _görüntü_kaydet(self):
        if self.işlenmiş_img is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek işlenmiş görüntü yok.")
            return
        yol = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if not yol:
            return
        if not self._görüntü_yaz(yol, self.işlenmiş_img):
            messagebox.showerror("Hata", "Görüntü kaydedilemedi.")
            return
        self.durum_var.set(f"Sonuç kaydedildi: {Path(yol).name}")

    def _gri_uygula(self):
        if not self._görüntü_kontrol():
            return
        sonuç = griye_cevir(self._orijinal_kopya())
        self._iki_panel_sonuç(sonuç, "1) Gri dönüşüm uygulandı.")

    def _binary_uygula(self):
        if not self._görüntü_kontrol():
            return
        eşik = int(self.binary_eşik.get())
        sonuç = binary_donusum(self._orijinal_kopya(), eşik)
        self._iki_panel_sonuç(sonuç, f"2) Binary dönüşüm uygulandı. Eşik: {eşik}")

    def _döndür_uygula(self):
        if not self._görüntü_kontrol():
            return
        açı = float(self.dönüş_açı.get())
        sonuç = goruntu_dondur(self._orijinal_kopya(), açı)
        self._iki_panel_sonuç(sonuç, f"3) Döndürme uygulandı. Açı: {açı:.0f}")

    def _kırp_uygula(self):
        if not self._görüntü_kontrol():
            return
        try:
            sonuç = goruntu_kirp(
                self._orijinal_kopya(),
                self.kırp_x.get(),
                self.kırp_y.get(),
                self.kırp_w.get(),
                self.kırp_h.get(),
            )
        except ValueError as hata:
            messagebox.showerror("Hata", str(hata))
            return
        self._iki_panel_sonuç(
            sonuç,
            (
                f"4) Kırpma uygulandı. Alan: "
                f"({self.kırp_x.get()}, {self.kırp_y.get()}) "
                f"{self.kırp_w.get()}x{self.kırp_h.get()}"
            ),
        )

    def _zoom_uygula(self):
        if not self._görüntü_kontrol():
            return
        ölçek = float(self.ölçek.get())
        yöntem = self.zoom_yöntem.get()
        sonuç = goruntu_olcekle(self._orijinal_kopya(), ölçek, yöntem)
        self._iki_panel_sonuç(sonuç, f"5) Zoom uygulandı. Ölçek: {ölçek:.1f}x | Yöntem: {yöntem}")

    def _renk_uzayı_uygula(self):
        if not self._görüntü_kontrol():
            return
        seçim = self.renk_uzayı.get()
        dönüşümler = {
            "RGB": bgr_to_rgb,
            "HSV": bgr_to_hsv,
            "YCrCb": bgr_to_ycrcb,
            "LAB": bgr_to_lab,
            "XYZ": bgr_to_xyz,
            "HLS": bgr_to_hls,
            "YUV": bgr_to_yuv,
            "LUV": bgr_to_luv,
        }
        sonuç = dönüşümler[seçim](self._orijinal_kopya())
        self._iki_panel_sonuç(sonuç, f"6) Renk uzayı dönüşümü uygulandı: {seçim}")

    def _histogram_eşitle(self):
        if not self._görüntü_kontrol():
            return
        gri = griye_cevir(self._orijinal_kopya())
        sonuç = histogram_esitle_gri(gri)
        self._iki_panel_sonuç(sonuç, "7) Histogram eşitleme uygulandı.")

    def _histogram_ger(self):
        if not self._görüntü_kontrol():
            return
        sonuç = histogram_gerdirme_renkli(self._orijinal_kopya())
        self._iki_panel_sonuç(sonuç, "7) Histogram germe uygulandı.")

    def _histogram_göster(self):
        if not self._görüntü_kontrol():
            return
        referans = self.işlenmiş_img
        if referans is None:
            referans = self.gürültülü_img if self.gürültülü_img is not None else self.orijinal_img
        fig = histogram_goster(self.orijinal_img, referans)
        pencere = tk.Toplevel(self.kök)
        pencere.title("Histogram")
        pencere.configure(bg=RENKLER["arka"])
        pencere.geometry("980x620")
        canvas = FigureCanvasTkAgg(fig, master=pencere)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)
        plt.close(fig)
        self.durum_var.set("7) Histogram penceresi açıldı.")

    def _aritmetik_uygula(self):
        if not self._görüntü_kontrol():
            return
        if self.ikinci_img is None:
            messagebox.showwarning("Uyarı", "Bu işlem için ikinci görüntüyü de yükleyin.")
            return
        işlem = self.aritmetik_işlem.get()
        sonuç = aritmetik_islem(self._orijinal_kopya(), self.ikinci_img, işlem)
        self._iki_panel_sonuç(sonuç, f"8) Aritmetik işlem uygulandı: {işlem}")

    def _parlaklık_uygula(self):
        if not self._görüntü_kontrol():
            return
        değer = int(self.parlaklık.get())
        sonuç = parlaklik_ayarla(self._orijinal_kopya(), değer)
        self._iki_panel_sonuç(sonuç, f"9) Parlaklık ayarlandı: {değer:+d}")

    def _kontrast_uygula(self):
        if not self._görüntü_kontrol():
            return
        oran = float(self.kontrast.get())
        sonuç = kontrast_ayarla(self._orijinal_kopya(), oran)
        self._iki_panel_sonuç(sonuç, f"9) Kontrast ayarlandı: {oran:.1f}x")

    def _gaussian_uygula(self):
        if not self._görüntü_kontrol():
            return
        boyut = int(self.gaussian_boyut.get())
        if boyut % 2 == 0:
            boyut += 1
        sigma = float(self.gaussian_sigma.get())
        sonuç = gaussian_blur(self._orijinal_kopya(), boyut, sigma)
        self._iki_panel_sonuç(
            sonuç,
            f"10) Gaussian uygulandı. Çekirdek: {boyut}x{boyut} | Sigma: {sigma:.1f}",
        )

    def _global_eşik_uygula(self):
        if not self._görüntü_kontrol():
            return
        eşik = int(self.eşik_global.get())
        sonuç = global_esikleme(self._orijinal_kopya(), eşik)
        self._iki_panel_sonuç(sonuç, f"11) Global eşikleme uygulandı. Eşik: {eşik}")

    def _adaptif_eşik_uygula(self):
        if not self._görüntü_kontrol():
            return
        blok = int(self.adaptif_blok.get())
        if blok % 2 == 0:
            blok += 1
        c_sabiti = int(self.adaptif_c.get())
        sonuç = adaptif_esikleme(self._orijinal_kopya(), blok, c_sabiti)
        self._iki_panel_sonuç(
            sonuç,
            f"11) Adaptif eşikleme uygulandı. Blok: {blok} | C: {c_sabiti}",
        )

    def _çift_eşik_uygula(self):
        if not self._görüntü_kontrol():
            return
        düşük = int(self.çift_düşük.get())
        yüksek = int(self.çift_yüksek.get())
        sonuç = cift_esikleme(self._orijinal_kopya(), düşük, yüksek)
        self._iki_panel_sonuç(
            sonuç,
            f"11) Çift eşikleme uygulandı. Düşük: {düşük} | Yüksek: {yüksek}",
        )

    def _sobel_uygula(self):
        if not self._görüntü_kontrol():
            return
        eşik = int(self.sobel_eşik.get())
        kenar, _, _ = sobel_kenar_bul(self._orijinal_kopya(), eşik)
        self._iki_panel_sonuç(kenar, f"12) Sobel uygulandı. Eşik: {eşik}")

    def _gürültü_ekle(self):
        if not self._görüntü_kontrol():
            return
        oran = float(self.gürültü_oran.get())
        gürültülü = salt_pepper_gurultu_ekle(self.orijinal_img.copy(), oran)
        self._üç_panel_sonuç(gürültülü, None, f"13) Salt & pepper gürültü eklendi. Oran: {oran:.2f}")

    def _mean_uygula(self):
        if not self._görüntü_kontrol():
            return
        gürültülü = self._gürültülü_kaynak()
        sonuç = mean_filtre(gürültülü.copy(), 3)
        self._üç_panel_sonuç(gürültülü, sonuç, "13) Mean filtre uygulandı (3x3).")

    def _median_uygula(self):
        if not self._görüntü_kontrol():
            return
        gürültülü = self._gürültülü_kaynak()
        sonuç = median_filtre(gürültülü.copy(), 3)
        self._üç_panel_sonuç(gürültülü, sonuç, "13) Median filtre uygulandı (3x3).")

    def _blur_uygula(self):
        if not self._görüntü_kontrol():
            return
        boyut = int(self.blur_boyut.get())
        if boyut % 2 == 0:
            boyut += 1
        tür = self.blur_tür.get()
        kaynak = self._orijinal_kopya()

        if tür == "ortalama":
            sonuç = ortalama_blur(kaynak, boyut)
        elif tür == "ağırlıklı":
            sonuç = agirlikli_ortalama_blur(kaynak, boyut)
        elif tür == "motion_yatay":
            sonuç = motion_blur(kaynak, boyut, "yatay")
        elif tür == "motion_dikey":
            sonuç = motion_blur(kaynak, boyut, "dikey")
        else:
            sonuç = motion_blur(kaynak, boyut, "capraz")

        self._iki_panel_sonuç(sonuç, f"14) Blur uygulandı: {tür} | Çekirdek: {boyut}x{boyut}")

    def _morfoloji_uygula(self):

        if not self._görüntü_kontrol():
            return

        işlem = self.morfo_işlem.get()
        boyut = int(self.morfo_boyut.get())
        tekrar = int(self.morfo_tekrar.get())

        if boyut % 2 == 0:
            boyut += 1

        binary = binary_donusum(self._orijinal_kopya(), 128)
        sonuç = morfolojik_islem(binary, işlem, boyut, tekrar)

        self._iki_panel_sonuç(
            sonuç,
            f"15) Morfolojik işlem uygulandı: {işlem} | Boyut: {boyut} | Tekrar: {tekrar}",
        )
        
if __name__ == "__main__":
    kök = tk.Tk()
    app = GörüntüİşlemeApp(kök)
    kök.mainloop()
