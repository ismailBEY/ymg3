# Derin Öğrenme Ödevi

CNN temelleri ve Fashion-MNIST sınıflandırması için ders raporu, sunum ve Python uygulaması.

## Dosyalar

- `Rapor.docx`: teorik yanıtlar, katman boyutları, yöntem, kaynakça ve sonuç tablosu.
- `Sunum_Teslim.pptx`: derste kullanılabilecek 8 slayt.
- `cnn_fashion_mnist.py`: konvolüsyon, pooling, ReLU/Softmax örnekleri ve iki CNN deneyini çalıştırır.

## Çalıştırma

Python 3.10+ ortamında:

```bash
pip install -r requirements.txt
python cnn_fashion_mnist.py
```

İlk çalıştırmada TensorFlow Fashion-MNIST veri kümesini indirir. `outputs/` içine pooling sonuç metni ve görseli, başlangıç filtrelerinin görseli, iki eğitim grafiği, deney ölçümleri ve eğitilmiş modeller kaydedilir.

Bu bilgisayardaki çalıştırma sonuçları: 8 filtrede eğitim doğruluğu %88,23, test doğruluğu %87,04 ve eğitim süresi 11,81 sn; 16 filtrede sırasıyla %88,72, %87,44 ve 12,58 sn. Çalıştırma ortamına göre küçük farklar oluşabilir; kendi ölçümlerinizi CSV ile karşılaştırıp rapor ve sunuma ad/grup bilgilerinizi ekleyin.

## Deney düzeni

İki koşulda ilk Conv2D katmanındaki filtre sayısı 8 ve 16 olarak değiştirilir. Diğer mimari ve eğitim ayarları aynıdır: ikinci katmanda 16 filtre, 5 epoch, batch size 64, Adam, %10 doğrulama bölümü. Sabit rastgele tohum başlangıç koşullarını karşılaştırılabilir kılar; süre ve son basamaklar donanım/yazılım sürümüne göre değişebilir.

## Kaynaklar

- [TensorFlow CNN tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [TensorFlow Fashion-MNIST classification tutorial](https://www.tensorflow.org/tutorials/keras/classification)
- [Keras Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/)
- [Keras MaxPooling2D](https://keras.io/api/layers/pooling_layers/max_pooling2d/)



