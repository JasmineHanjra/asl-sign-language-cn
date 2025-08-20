[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Convolutional Neural Network for American Sign Language (ASL) Recognition

## 📌 Project Overview
This project was developed as part of a group effort to build a **Convolutional Neural Network (CNN)** capable of recognizing American Sign Language (ASL) gestures and converting them into text. The aim was to create a system that bridges communication between signers and non-signers, supporting accessibility for the deaf and hard-of-hearing community.

The model was trained on both publicly available datasets and a private dataset (not shareable). By expanding the dataset and tuning the model architecture, we achieved a significant performance improvement while ensuring robustness across diverse gestures.

---

## 🎯 Goals
- Build a CNN that accurately classifies ASL gestures from images.  
- Improve model performance through architectural adjustments.  
- Enable real-time prediction of ASL gestures into text.  
- Explore scalability for dynamic gestures and real-world applications.  

---

## 📊 Dataset
- **Initial Dataset:** 24,000 images (ASL alphabet A–Z).  
- **Expanded Dataset:** 203,000 images covering:
  - Letters (A–Z)  
  - Numbers (0–9)  
  - Common words (e.g., *Help*, *Friend*, *Stop*):contentReference[oaicite:0]{index=0}  
- **Private Dataset:** An additional dataset was used during training but cannot be shared publicly due to access restrictions.  
<img width="1050" height="687" alt="image" src="https://github.com/user-attachments/assets/cce017c2-eaeb-40b6-80fa-a5a9e2dd7fd1" />

Public dataset link: [Kaggle – American Sign Language Dataset](https://www.kaggle.com/datasets/alhasangamalmahmoud/american-sign-language-asl)  

---

## 🛠️ Tools and Technologies
- **Programming Language:** Python  
- **Deep Learning Libraries:** TensorFlow, Keras, PyTorch, Scikit-learn  
- **Environment:** Jupyter Notebook, Kaggle (GPU acceleration)  

---

## 🔬 Methodology
1. **Data Preprocessing:**  
   - Normalized and resized images to 64×64 pixels.  
   - Applied augmentation techniques (rotation, flipping) to address imbalance and variability.  

2. **Model Training & Modifications:**  
   - Built upon an existing CNN architecture.  
   - Adjusted dropout rates to balance overfitting and generalization.  
   - Increased convolutional filters to capture complex patterns.  
   - Implemented **early stopping** (patience = 5 epochs) to prevent overtraining:contentReference[oaicite:1]{index=1}.  

3. **Evaluation:**  
   - Accuracy, precision, recall, F1-score.  
   - Confusion matrix analysis (most misclassifications occurred with visually similar gestures like "O" vs "0").  

---

## 📈 Results
- **Before Modifications:** 96.4% accuracy  
- **After Modifications:** 97.8% accuracy:contentReference[oaicite:2]{index=2}  
- Demonstrated the effectiveness of tuning CNN hyperparameters for better performance.  

---

## 🚀 Features
- Upload an ASL gesture image and receive the predicted letter, number, or word as text output.  
- Real-time prediction support for educational and accessibility use.  

---

## 🔮 Future Work
- Extend recognition to **dynamic gestures** using RNNs or Transformers.  
- Deploy lightweight models for mobile/AR applications.  
- Partner with deaf community organizations for real-world testing and feedback.  
- Expand to other sign languages (e.g., BSL, ISL).  

---

## 👥 Contributors
- Ilsa Qadir  
- Krishma Kapoor  
- Jasmine Hanjra  

---

## 📂 Documentation
Can be found under docs
- [Project Proposal]
- [Final Presentation]


## 📚 References
- Kaggle Dataset: [American Sign Language (ASL)](https://www.kaggle.com/datasets/alhasangamalmahmoud/american-sign-language-asl)  
- Kaggle Notebook (baseline model): [ASL Model](https://www.kaggle.com/code/raanaramadan/asl-model)  
- Modified Model: [Deep Learning Project – ASL to Text CNN](https://www.kaggle.com/code/ilsaqadir/deep-learning-project-asl-to-text-cnn)  

---
