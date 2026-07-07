# 🔢 Handwritten Number Recognition (PyTorch)

An end-to-end Deep Learning project designed to recognize and predict handwritten digits using PyTorch. This project implements and compares two different neural network architectures—Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN)—to demonstrate the power of spatial feature extraction.

---

## 📊 Model Performance & Results

- **ANN Model:** Achieved a stellar **96% accuracy** by flattening the pixel data and processing it through fully connected layers.
- **CNN Model:** Achieved a superior **98% accuracy** by leveraging convolutional and pooling layers to capture spatial features and patterns effectively.

---

## 🚀 Key Features

- **PyTorch Framework:** Built entirely using PyTorch's `torch.nn` and `torch.optim` modules.
- **Dual Architecture:** Offers a direct performance comparison between standard dense networks (ANN) and computer vision standards (CNN).
- **High Reliability:** Optimized hyperparameter tuning and data normalization result in up to 98% prediction precision.

---

## 📁 Project Structure

- `ann_model.pth` / `ann_model.pickle`: Main weights and architecture for the trained ANN model.
- `cnn_model.pth` / `cnn_model.pickle`: Main weights and architecture for the trained CNN model.
- `train.py` / `notebook.ipynb`: Core code containing data loading, preprocessing, model definitions, and training loops.
- `README.md`: Detailed documentation of the project.

---

## 🛠️ How It Works

1. **Data Preprocessing:** Input digit images are normalized and converted into PyTorch tensors.
2. **Feature Extraction:** \* **ANN** flattens the image matrix into a 1D vector.
   - **CNN** applies convolutional filters directly onto the 2D image matrix to retain spatial relationships.
3. **Inference:** The models output a probability distribution across 10 classes (digits 0-9), selecting the highest probability as the final predicted number.

---

## 💻 Tech Stack

- **Framework:** PyTorch
- **Language:** Python
- **Libraries:** Torchvision, NumPy, Matplotlib

---

## 📜 License

This project is licensed under the MIT License.

A deep learning project utilizing PyTorch to perform handwritten number recognition. It implements both Artificial Neural Network (ANN) and Convolutional Neural Network (CNN) architectures. The ANN model achieves an impressive 96% accuracy, while the optimized CNN model delivers a superior performance with 98% accuracy.
