"""CNN temelleri ve Fashion-MNIST deneyleri.

Kurulum: pip install tensorflow matplotlib numpy
Çalıştırma: python cnn_fashion_mnist.py
İki konfigürasyonu aynı koşullarda eğitir; sonuçları outputs/ içine yazar.
"""
from pathlib import Path
import csv
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
CLASS_NAMES = ["Tişört/üst", "Pantolon", "Kazak", "Elbise", "Ceket",
               "Sandalet", "Gömlek", "Spor ayakkabı", "Çanta", "Bot"]


def inspect_convolution():
    """Ödevde istenen 128x128 örnek girdi ve 16 filtreli Conv2D."""
    tf.keras.utils.set_random_seed(42)
    conv = tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation="relu",
        kernel_initializer="he_normal"
    )
    dummy_input = tf.random.normal((1, 128, 128, 1))
    output = conv(dummy_input)
    weights, biases = conv.get_weights()
    print("Giriş boyutu:", dummy_input.shape)
    print("Çıkış boyutu:", output.shape)
    print("Filtre ağırlıklarının boyutu:", weights.shape)
    print("Bias boyutu:", biases.shape)
    print("İlk filtre:\n", weights[:, :, 0, 0])

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        kernel = weights[:, :, 0, i]
        lim = np.max(np.abs(kernel)) or 1
        ax.imshow(kernel, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(f"Filtre {i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Başlangıçta rastgele başlatılan 3×3 filtre ağırlıkları")
    fig.tight_layout()
    fig.savefig(OUT / "baslangic_filtreleri.png", dpi=180)
    plt.close(fig)


def inspect_pooling():
    values = np.array([[1., 3., 2., 4.], [5., 6., 1., 2.],
                       [7., 2., 8., 3.], [4., 1., 5., 9.]])
    data = tf.reshape(tf.constant(values), (1, 4, 4, 1))
    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    result = pool(data).numpy()[0, :, :, 0]
    print("Pooling girişi:\n", values)
    print("Pooling sonucu:\n", result)
    # Elle beklenen sonuç: [[6, 4], [7, 9]]
    assert np.array_equal(result, np.array([[6., 4.], [7., 9.]]))
    (OUT / "pooling_sonuclari.txt").write_text(
        "Girdi matrisi (4x4):\n"
        f"{values}\n\n"
        "2x2 Max Pooling, stride=2 sonucu (2x2):\n"
        f"{result}\n",
        encoding="utf-8",
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, matrix, title in zip(axes, (values, result), ("Girdi 4×4", "Max Pooling çıktısı 2×2")):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=9)
        ax.set_title(title)
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                color = "white" if value >= 6 else "#172B3A"
                ax.text(col, row, f"{value:g}", ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(OUT / "pooling_girdisi_cikisi.png", dpi=180)
    plt.close(fig)


def inspect_activations():
    """ReLU ve Softmax örneklerini sayısal olarak göster."""
    x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
    print("ReLU girişi:", x.numpy())
    print("ReLU çıkışı:", tf.keras.layers.ReLU()(x).numpy())
    logits = tf.constant([[2.0, 1.0, 0.1]])
    probabilities = tf.keras.layers.Softmax()(logits)
    print("Softmax sonuçları:", probabilities.numpy())
    print("Olasılıkların toplamı:", tf.reduce_sum(probabilities).numpy())


def build_model(first_filters: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(first_filters, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


def plot_history(history, label):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="Eğitim")
    axes[0].plot(history.history["val_accuracy"], label="Doğrulama")
    axes[0].set_title("Doğruluk")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Eğitim")
    axes[1].plot(history.history["val_loss"], label="Doğrulama")
    axes[1].set_title("Kayıp")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(f"İlk katmanda {label} filtre")
    fig.tight_layout()
    fig.savefig(OUT / f"egitim_grafikleri_{label}_filtre.png", dpi=180)
    plt.close(fig)


def main():
    inspect_convolution()
    inspect_pooling()
    inspect_activations()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    print("Eğitim verisi:", x_train.shape, "Etiket:", y_train.shape)
    print("Test verisi:", x_test.shape, "Etiket:", y_test.shape)

    rows = []
    for first_filters in (8, 16):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = build_model(first_filters)
        model.summary()
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        start = time.perf_counter()
        history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                            validation_split=0.1, verbose=2)
        elapsed = time.perf_counter() - start
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        train_accuracy = history.history["accuracy"][-1]
        plot_history(history, first_filters)
        model.save(OUT / f"fashion_mnist_{first_filters}_filtre.keras")
        rows.append({"ilk_katman_filtre": first_filters,
                     "egitim_dogrulugu": train_accuracy,
                     "test_dogrulugu": test_accuracy,
                     "test_kaybi": test_loss,
                     "egitim_suresi_saniye": elapsed})
        print(f"{first_filters} filtre | eğitim doğruluğu={train_accuracy:.4f} "
              f"| test doğruluğu={test_accuracy:.4f} | test kaybı={test_loss:.4f} "
              f"| süre={elapsed:.1f} sn")

    with (OUT / "deney_sonuclari.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Sonuç dosyası:", OUT / "deney_sonuclari.csv")
    print("Sınıf adları:", CLASS_NAMES)


if __name__ == "__main__":
    main()
