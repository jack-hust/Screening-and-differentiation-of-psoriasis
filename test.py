import tensorflow as tf
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Thư mục chứa ảnh
data_dir = {
    "Vảy nến không mủ": r"D:\AI vay nen\vaynen\12",
    "Vảy nến có mủ": r"D:\AI vay nen\vaynen\13"
}

# Định nghĩa kích thước ảnh chuẩn của MobileNetV2
img_size = (224, 224)

# Load mô hình MobileNetV2 để trích xuất đặc trưng
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Mảng chứa đặc trưng và nhãn
features = []
labels = []

for label, folder in data_dir.items():
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        try:
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  # Chuẩn hóa ảnh theo MobileNetV2

            # Trích xuất đặc trưng
            feature_vector = base_model.predict(img_array)

            features.append(feature_vector[0])  # Lưu đặc trưng
            labels.append(1 if label == "Vảy nến có mủ" else 0)  # 1: có mủ, 0: không mủ
        except:
            print(f"Lỗi đọc ảnh: {img_path}")

# Chuyển thành mảng numpy
X = np.array(features)
y = np.array(labels)

# Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Độ chính xác: {accuracy:.2f}")

# Lưu mô hình đã huấn luyện
joblib.dump(classifier, "psoriasis_classifier.pkl")


def predict_image(img_path, base_model, classifier):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Trích xuất đặc trưng
    feature_vector = base_model.predict(img_array)

    # Dự đoán nhãn
    prediction = classifier.predict(feature_vector)[0]

    # Hiển thị kết quả
    label = "Vảy nến có mủ" if prediction == 1 else "Vảy nến không mủ"
    plt.imshow(img)
    plt.title(f"Dự đoán: {label}")
    plt.axis("off")
    plt.show()


# Kiểm tra với ảnh mới
# predict_image(r"D:\Vảy nến\test_image.jpg", base_model, classifier)
