import os

import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
keras_file_path = os.path.join(BASE_DIR, "dob app", "app", "model", "12meiv1.keras")
saved_model_path = 'saved_model_12meiv1'

print(f"Mengecek keberadaan file: {keras_file_path}")
if os.path.exists(keras_file_path):
    print(f"File ditemukan di: {os.path.abspath(keras_file_path)}")
    try:
        model = tf.keras.models.load_model(keras_file_path)
        model.export(saved_model_path)  # Hapus argumen save_format
        print(f"Model berhasil disimpan dalam format SavedModel di direktori: {saved_model_path}")
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat/menyimpan model: {e}")
else:
    print(f"File tidak ditemukan di path: {os.path.abspath(keras_file_path)}")