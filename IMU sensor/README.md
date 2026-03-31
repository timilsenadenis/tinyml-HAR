# 🩸 AnemiaVision — Non-Invasive Anemia Detection

> Detect anemia severity from eye conjunctiva photos using deep learning.
> No blood test needed — just a phone camera.

---

## 📌 Project Overview

AnemiaVision is an AI-powered anemia screening tool that analyzes the **palpebral conjunctiva** (inner lower eyelid) for pallor — a key clinical indicator of anemia. It classifies severity into 4 levels using a **MobileNetV2** transfer learning model.

### Output Classes
| Class | Hemoglobin | Color |
|-------|-----------|-------|
| Non-Anemic | Hb > 12 g/dL | 🟢 |
| Mild | Hb 8–12 g/dL | 🟡 |
| Moderate | Hb 5–8 g/dL | 🟠 |
| Severe | Hb < 5 g/dL | 🔴 |

---

## 📁 Project Structure

```
anemia-detection/
├── data/
│   ├── raw/                        # Original downloaded datasets
│   │   ├── eyes_defy_anemia/       # Dataset 1 (Italy/ + India/)
│   │   ├── palpebral_conjunctiva/  # Dataset 2
│   │   └── anemia_detection/       # Dataset 3
│   ├── processed/                  # Labeled + preprocessed images
│   │   ├── Non_Anemic/
│   │   ├── Mild/
│   │   ├── Moderate/
│   │   └── Severe/
│   └── split/                      # Train/Val/Test splits
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── data_preparation.py         # Hb → severity label mapping
│   ├── preprocessing.py            # CLAHE + augmentation + split
│   ├── model.py                    # MobileNetV2 architecture
│   ├── train.py                    # Training script (2-phase)
│   └── predict.py                  # Inference module
├── app/
│   ├── app.py                      # Streamlit application (main)
│   ├── pdf_report.py               # PDF report generator
│   └── history.py                  # SQLite history tracking
├── model/
│   ├── anemia_model.h5             # Trained model (generated)
│   └── anemia_model.tflite         # TFLite mobile model (generated)
├── reports/                        # Generated PDF reports
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### Step 1: Clone & Install
```bash
git clone <your-repo>
cd anemia-detection
pip install -r requirements.txt
```

### Step 2: Download Datasets from Kaggle
Download these 3 datasets and place them in `data/raw/`:

| Dataset | Kaggle URL | Place in |
|---------|-----------|----------|
| Eyes-Defy-Anemia | kaggle.com/datasets/harshwardhanfartale/eyes-defy-anemia | `data/raw/eyes_defy_anemia/` |
| Palpebral Conjunctiva | kaggle.com/datasets/guptajanavi/palpebral-conjunctiva-to-detect-anaemia | `data/raw/palpebral_conjunctiva/` |
| Anemia Detection | kaggle.com/datasets/shahriar26s/anemia-detection-dataset | `data/raw/anemia_detection/` |

### Step 3: Prepare Data
```bash
python src/data_preparation.py
```

### Step 4: Preprocess & Augment
```bash
python src/preprocessing.py
```

### Step 5: Train Model
```bash
python src/train.py
```
> 💡 Use Google Colab with GPU for faster training.

### Step 6: Run the App
```bash
streamlit run app/app.py
```

---

## 📱 Live Demo Flow

```
User pulls lower eyelid
        ↓
Takes clear photo with phone camera
        ↓
Uploads to AnemiaVision app
        ↓
CLAHE enhancement applied
        ↓
MobileNetV2 predicts severity
        ↓
Shows result + recommendations + PDF report
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
    → Data Augmentation (flip, rotate, brightness, zoom)
    → MobileNetV2 (ImageNet pretrained, frozen)
    → GlobalAveragePooling2D
    → Dense(256, ReLU) + BatchNorm + Dropout(0.4)
    → Dense(128, ReLU) + BatchNorm + Dropout(0.2)
    → Dense(4, Softmax)
```

**Training Strategy:**
- Phase 1: Frozen MobileNetV2 (30 epochs, LR=1e-3)
- Phase 2: Fine-tune top 30 layers (20 epochs, LR=1e-5)

---

## 📊 App Features

| Feature | Description |
|---------|-------------|
| 🔍 Detection | Upload conjunctiva photo → get severity |
| 📊 Confidence Chart | Probability bar chart for all classes |
| 💊 Recommendations | Tailored medical advice per severity |
| 📈 History Tracking | SQLite-based scan history per patient |
| 📄 PDF Report | Downloadable professional report |

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**.
It is **NOT a certified medical device** and should not replace professional medical diagnosis.
Always consult a qualified healthcare provider for anemia diagnosis and treatment.

---

## 👨‍💻 Tech Stack

- **Model**: TensorFlow / Keras / MobileNetV2
- **Image Processing**: OpenCV (CLAHE)
- **App**: Streamlit + Plotly
- **PDF**: ReportLab
- **Database**: SQLite
- **Deployment**: Streamlit Cloud / Google Colab + ngrok
