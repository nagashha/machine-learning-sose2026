# Makine Öğrenmesi - Ders Çalışmaları (Yaz Dönemi 2026)

🇬🇧 [English version → README.md](README.md)

Bu repo, Göttingen Üniversitesi'ndeki **Machine Learning** dersinde (2026 yaz dönemi)
üç kişilik grubumuzla yaptığımız çalışmaları içeriyor. Elle yazılmış basit doğrusal
regresyondan başlayıp PyTorch ile sinir ağlarına, en sonda da gerçek Airbnb verisiyle
fiyat tahmini yapan bir projeye kadar gidiyor.

## Kısaca içerik

| # | Klasör | Ne yapıyor? | Sonuç |
|---|---|---|---|
| 1 | `practicals/01-linear-regression` | Süperiletken malzemelerin kritik sıcaklığını tahmin etme | Daha fazla veriyle R² 0.52 → 0.73 |
| 2 | `practicals/02-pytorch-linear-regression` | Aynı problem, bu sefer PyTorch ile | Doğrulama MSE ≈ 340 (Practical 1 ile aynı seviye) |
| 3 | `practicals/03-image-classification-cifar10` | Küçük renkli resimleri 10 sınıfa ayırma | Doğruluk %39 → %67 |
| ★ | `airbnb-price-prediction` | **Final projesi:** Dublin'deki Airbnb evlerinin gecelik fiyatını tahmin etme | Ortalama hata gecelik €41 (basit tahminde €94), R² 0.72 |

---

## 1. Practical 1 - Doğrusal Regresyon

**Problem:** Elimizde ~21.000 süperiletken malzeme ve her biri için 81 fiziksel/kimyasal
özellik var. Amaç, malzemenin süperiletken hale geldiği **kritik sıcaklığı** tahmin etmek.

**Ne yaptık?**
- Veriyi grafiklerle inceledik ve normalize ettik (her özelliği ortalama 0, standart sapma 1 olacak şekilde ölçekledik).
- **En küçük kareler (OLS)** yöntemini formülüyle, hazır kütüphane kullanmadan NumPy ile kendimiz yazdık; sonra scikit-learn ile aynı sonucu aldığımızı doğruladık.
- **Ridge regresyon** ekledik: modelin ağırlıklarını küçük tutmaya zorlayan bir "ceza" terimi. Bu, modelin eğitim verisini ezberlemesini (overfitting) engeller. Ceza gücü α'yı çok küçükten çok büyüğe kadar deneyip etkisini çizdik.
- α'nın en iyi değerini test verisine bakmadan bulmak için **çapraz doğrulamayı (cross-validation)** sıfırdan yazdık.

**Sonuç:** Sadece 200 örnekle eğitince model ezberliyor (R² 0.52), ridge bunu iyileştiriyor
(R² 0.66). 20.000 örnekle eğitince R² 0.73'e çıkıyor ve neredeyse hiç cezaya gerek kalmıyor -
yani çok veri, en iyi düzenlileştirme (regularization) yöntemi.

## 2. Practical 2 - PyTorch'a Giriş

**Problem:** Practical 1'deki aynı veri, ama bu sefer model PyTorch ile kuruluyor.
Doğrusal regresyon aslında tek nöronlu bir sinir ağı olduğu için, sinir ağlarının
eğitim mantığını öğrenmek için iyi bir başlangıç.

**Ne yaptık?**
- PyTorch tensörlerini tanıdık (NumPy ile ilişkisi, CPU/GPU).
- Kendi `Dataset` sınıfımızı ve `DataLoader`'ları yazdık; veriyi eğitim / doğrulama / test olarak üçe böldük.
- Gradyan inişini (gradient descent) önce elle, sonra PyTorch'un `SGD` optimizer'ı ile yaptık.
- Eğitim ve doğrulama kayıplarını (loss) her epoch için kaydedip çizdik.
- **Öğrenme hızının (learning rate)** etkisini denedik: çok küçükse yavaş öğreniyor, çok büyükse model "patlıyor" (kayıp sonsuza gidiyor).
- **Early stopping** ekledik: doğrulama kaybı birkaç epoch boyunca iyileşmezse eğitimi durduruyor.
- **Batch size**'ın etkisini inceledik.

**Sonuç:** Uygun öğrenme hızıyla model, Practical 1'deki formül çözümüyle aynı hata
seviyesine (MSE ≈ 340) ulaşıyor. Öğrenme hızı 0.1 ve üstünde kayıp patlıyor. Küçük
batch'ler daha az epoch'ta öğreniyor çünkü her epoch'ta çok daha fazla güncelleme yapılıyor.

## 3. Practical 3 - CIFAR-10 Görüntü Sınıflandırma

**Problem:** 32x32 piksellik 60.000 renkli resmi 10 sınıfa ayırmak (uçak, otomobil, kuş,
kedi, geyik, köpek, kurbağa, at, gemi, kamyon).

**Ne yaptık?** Aynı problemi giderek güçlenen dört modelle çözüp karşılaştırdık:

| Model | Basitçe ne? | Test doğruluğu |
|---|---|---|
| Lojistik regresyon | Pikselleri tek bir doğrusal katmana veriyor | ≈ %39 |
| MLP | 4 katmanlı klasik sinir ağı | ≈ %56 |
| MLP + L2 | Aynı ağ, ağırlıklara ceza eklenmiş | ≈ %56 |
| **LeNet-5 (CNN)** | Resimdeki yerel desenleri (kenar, şekil) yakalayan evrişimli ağ | **≈ %67** |

**Sonuç:** Evrişimli ağ (CNN), MLP'den ~100 kat daha az parametreyle en iyi sonucu
veriyor; çünkü resimlerde yan yana piksellerin birbiriyle ilişkili olduğunu kullanıyor.
Karışıklık matrisinde en çok kedi ↔ köpek ve otomobil ↔ kamyon karıştırılıyor.

## ★ Final Projesi - Airbnb Fiyat Tahmini (Dublin)

**Problem:** [Inside Airbnb](https://insideairbnb.com/) sitesinden alınan Dublin verisiyle
(Eylül 2025, ~7.000 ilan) bir evin **gecelik fiyatını** tahmin etmek.

**Ne yaptık?** Hocanın istediği gibi en az iki veri türü (modalite) kullandık - biz üç tane kullandık:
- **Tablo verisi:** oda sayısı, kaç kişilik, oda tipi, ev sahibi bilgileri, yorum puanları, olanaklar (wifi, mutfak, jakuzi...) vb.
- **Metin:** ilanın başlığı ve açıklaması → TF-IDF ile sayılara çevrilip sıkıştırıldı.
- **Konum:** enlem/boylam, şehir merkezine uzaklık ve KMeans ile bulunan 12 "mini-mahalle".

Üç modeli (Ridge, Random Forest, Gradient Boosting) bu veri türlerinin dört farklı
kombinasyonuyla 5 katlı çapraz doğrulamayla karşılaştırdık, en iyisini ayarladık
(hyperparameter tuning) ve en sonda **bir kez** test verisinde ölçtük.

**Sonuç:**
- En iyi model (Gradient Boosting + tablo + konum verisi) gecelik fiyatı **ortalama €41
  hatayla** tahmin ediyor; ilanların yarısında hata €21'in altında. Herkese ortanca fiyatı
  söyleyen basit tahminin hatası €94 - yani model hatayı yarıdan fazla azaltıyor.
- **Konum** her modele yardımcı oluyor: merkezden uzaklaştıkça fiyat düşüyor.
- **Metin** basit doğrusal modele çok yardımcı oluyor, ağaç tabanlı modellere neredeyse hiç -
  çünkü açıklamadaki bilgilerin çoğu (kaç kişilik, oda tipi, olanaklar) zaten tabloda var.
- Fiyatı en çok belirleyenler: kaç kişilik olduğu, oda tipi, merkeze uzaklık ve yatak odası sayısı.
- Hiperparametre ayarı (tuning) bu sefer sonucu iyileştirmedi; varsayılan ayarlar zaten iyiydi.

Kod tek bir notebook yerine adım adım çalışan script'lere bölündü
(`data_prep → eda → train → tune → evaluate`). Ayrıntılar:
[`airbnb-price-prediction/README.md`](airbnb-price-prediction/README.md).

---

## Nasıl çalıştırılır?

```bash
git clone <bu-repo>
cd machine-learning-sose2026
pip install -r requirements.txt
```

- **Practical 1-2:** Süperiletken verisini [`practicals/data/README.md`](practicals/data/README.md)'deki linkten indirip `practicals/data/superconduct_train.csv` olarak koyun, sonra notebook'ları `jupyter lab` ile açın.
- **Practical 3:** CIFAR-10 otomatik iniyor; GPU önerilir (Kaggle / Colab).
- **Airbnb projesi:** Repoda küçük bir örnek veri var, tam veri için `airbnb-price-prediction/data/README.md`'ye bakın.

## Ekip

Muhammed Nagaş · Felix Fischer · Niclas Langer

Practical notebook'larındaki görev metinleri ve bazı yardımcı kodlar dersin
hocalarına aittir; çözümler ve cevaplar bize aittir.
