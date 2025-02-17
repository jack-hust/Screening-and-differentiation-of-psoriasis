import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split

# Định nghĩa thư mục chứa ảnh
data_dir = r"D:\AI vay nen\vaynen"  # Thư mục gốc chứa 2 thư mục con: 12 (không mủ), 13 (có mủ)

# Kích thước ảnh đầu vào của EfficientNetB0
img_size = (224, 224)
batch_size = 16

# Load dữ liệu từ thư mục (Data Augmentation)
datagen = ImageDataGenerator(
    rescale=1.0 / 255,        # Chuẩn hóa giá trị pixel về [0, 1]
    rotation_range=30,        # Xoay ảnh tối đa 30 độ
    width_shift_range=0.2,    # Dịch ảnh theo chiều ngang
    height_shift_range=0.2,   # Dịch ảnh theo chiều dọc
    shear_range=0.2,          # Biến dạng ảnh
    zoom_range=0.2,           # Phóng to/thu nhỏ
    horizontal_flip=True,     # Lật ngang ảnh
    validation_split=0.2      # 20% dữ liệu dành cho validation
)

# Load ảnh cho tập huấn luyện
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

# Load ảnh cho tập validation
val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# Load mô hình EfficientNetB0 pre-trained
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Đóng băng các lớp của mô hình gốc
base_model.trainable = False

# Thêm các lớp mới để huấn luyện
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)  # 1 output vì bài toán nhị phân (có mủ/không mủ)

# Tạo mô hình mới
model = Model(inputs=base_model.input, outputs=output)

# Compile mô hình
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Huấn luyện mô hình
epochs = 89
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=1
)

# Lưu mô hình sau khi huấn luyện
model.save("psoriasis_efficientnetb0.h5")

val_loss, val_acc = model.evaluate(val_generator)
print(f" Độ chính xác trên tập validation: {val_acc:.2f}")

def predict_image(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) / 255.0  # Chuẩn hóa pixel
    img_array = np.expand_dims(img, axis=0)  # Thêm batch dimension

    # Dự đoán
    prediction = model.predict(img_array)[0][0]
    label = "Vảy nến có mủ" if prediction >= 0.5 else "Vảy nến không mủ"

    # Hiển thị ảnh và kết quả
    plt.imshow(img)
    plt.title(f"Dự đoán: {label}")
    plt.axis("off")
    plt.show()

# Kiểm tra với ảnh mới
#model = tf.keras.models.load_model("psoriasis_efficientnetb0.h5")
#predict_image(r"D:\Vảy nến\test_image.jpg", model)

