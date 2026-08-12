# 🌱 AI-Based Crop Recommendation System
🔗 **Live Demo:** https://ai-crop-recommendation-system-chetan.streamlit.app/
An end-to-end machine learning application that recommends a suitable crop based on soil and environmental conditions.

The project uses a **Tuned Random Forest classifier** trained on soil nutrients and environmental measurements. A **Streamlit** web application provides an easy-to-use interface for farmers, students, and other users to enter measurements and receive a crop recommendation.

---

## 🚀 Project Overview

Choosing a suitable crop depends on several factors such as:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

This project uses these seven features to predict one of **22 crop classes**.

### 🎯 Problem Statement

> **AI-Based Crop Recommendation for Farmers**

The objective is to build a machine learning system that can recommend an appropriate crop from soil and environmental measurements.

---

## ✨ Features

### 🤖 Machine Learning
- Tuned Random Forest classification model
- 22 crop classes
- 7 input features
- Stratified train/test split
- Cross-validation
- Model comparison
- Feature importance analysis
- Classification report
- Confusion matrix
- Error analysis

### 🌱 Streamlit Application
- Professional two-column input layout
- Soil and environmental input fields
- Crop recommendation
- Model confidence
- Top-3 crop recommendations
- Probability visualization
- Crop information
- Input summary
- Farmer-friendly warnings
- Exact model input preview
- Prediction history
- Reset inputs
- Downloadable prediction report
- About-model section
- Future-features section
- Deployment-ready structure

---

## 📊 Dataset

The dataset contains **2,200 records** and **8 columns**.

### Input Features

| Feature | Description |
|---|---|
| `N` | Nitrogen content in soil |
| `P` | Phosphorus content in soil |
| `K` | Potassium content in soil |
| `temperature` | Temperature in °C |
| `humidity` | Relative humidity in % |
| `ph` | Soil pH |
| `rainfall` | Rainfall in mm |

### Target

| Column | Description |
|---|---|
| `label` | Recommended crop |

### Dataset Summary

- Total samples: **2,200**
- Input features: **7**
- Crop classes: **22**
- Missing values: **0**
- Duplicate rows: **0**
- Samples per crop: **100**

### Crop Classes

The dataset contains:

1. Apple
2. Banana
3. Blackgram
4. Chickpea
5. Coconut
6. Coffee
7. Cotton
8. Grapes
9. Jute
10. Kidneybeans
11. Lentil
12. Maize
13. Mango
14. Mothbeans
15. Mungbean
16. Muskmelon
17. Orange
18. Papaya
19. Pigeonpeas
20. Pomegranate
21. Rice
22. Watermelon

---

## 🔎 Exploratory Data Analysis

The dataset was examined before model training.

### Data Quality

- All 2,200 records contain values for every feature.
- No duplicate rows were found.
- There are 22 crop classes.
- Each crop contains 100 samples.

### Important Observations

- Nitrogen distributions vary across crops.
- Phosphorus and potassium show the strongest positive correlation among the measured features.
- Rainfall has a comparatively wide range.
- Temperature, humidity, pH, rainfall, and nutrient levels show different distributions across crops.
- Several features have overlapping distributions, so crop prediction benefits from using multiple features together rather than relying on a single measurement.

### Domain Consideration

A value such as `N = 0` was **not blindly removed** because an apparently low nutrient value should be evaluated using agricultural/domain knowledge before treating it as an invalid observation.

---

## 🧪 Train/Test Split

The dataset was separated into features and target:

```text
X shape: (2200, 7)
y shape: (2200,)
```

The data was split into:

```text
Training samples: 1760
Testing samples: 440
```

A **stratified split** was used so that the crop distribution remained balanced between the training and testing sets.

Each crop had:

```text
80 training samples
20 testing samples
```

---

## 🤖 Model Development

Several classification algorithms were evaluated.

| Model | Test Accuracy |
|---|---:|
| Decision Tree | 97.95% |
| KNN | 97.95% |
| Logistic Regression | 97.27% |
| Random Forest | **99.55%** |

KNN and Logistic Regression are sensitive to feature scale, so scaling is relevant when using those models. Tree-based models such as Random Forest do not require feature scaling in the same way.

---

## 🏆 Final Model

The final model is a **Tuned Random Forest Classifier**.

### Best Parameters

```python
{
    'max_depth': 10,
    'min_samples_leaf': 1,
    'min_samples_split': 5,
    'n_estimators': 100
}
```

### Performance

```text
Cross-validation accuracy: 99.6023%

Test accuracy: 99.5455%

Test samples: 440
Correct predictions: 438
Incorrect predictions: 2
```

The tuned model achieved the same test accuracy as the original Random Forest on the held-out test set, while cross-validation was used to verify the model during tuning.

---

## 📈 Classification Performance

The final model performed very consistently across the 22 crop classes.

Most classes achieved:

```text
Precision: 1.00
Recall:    1.00
F1-score:  1.00
```

The small number of errors occurred between:

```text
Actual: blackgram → Predicted: maize
Actual: rice      → Predicted: jute
```

The two incorrect test samples were:

```text
Blackgram:
N = 60
P = 59
K = 22
Temperature = 31.868473
Humidity = 66.742175
pH = 7.191523
Rainfall = 74.222386

Predicted: maize
```

```text
Rice:
N = 67
P = 43
K = 39
Temperature = 26.043720
Humidity = 84.969072
pH = 5.999969
Rainfall = 186.753677

Predicted: jute
```

These examples show why some crops can be difficult to distinguish when their feature values overlap.

---

## 🌟 Feature Importance

Feature importance from the Random Forest showed:

| Feature | Importance |
|---|---:|
| Rainfall | 0.230184 |
| pH | 0.050608 |

Rainfall was the highest-importance feature in the observed model output, while pH had the lowest importance.

The low importance of a feature does **not** automatically mean it should be removed. A feature may still contribute useful information when combined with other variables.

The seven features were therefore retained:

```text
N
P
K
temperature
humidity
ph
rainfall
```

---

## 💻 Application Architecture

The project follows this basic flow:

```text
User
  │
  ▼
Streamlit Web Interface
  │
  ├── N
  ├── P
  ├── K
  ├── Temperature
  ├── Humidity
  ├── Soil pH
  └── Rainfall
  │
  ▼
Input Validation
  │
  ▼
Feature Ordering
  │
  ▼
Tuned Random Forest Model
  │
  ├── Recommended Crop
  ├── Prediction Probability
  └── Top-3 Recommendations
  │
  ▼
Results + Crop Information + Report
```

---

## 📁 Project Structure

```text
AI-Crop-Recommendation-System/
│
├── app.py
│
├── crop_recommendation_model.pkl
├── crop_feature_names.pkl
│
├── requirements.txt
├── README.md
│
└── .gitignore
```

### Important Files

#### `app.py`

The main Streamlit application.

#### `crop_recommendation_model.pkl`

The trained/tuned Random Forest model.

#### `crop_feature_names.pkl`

Stores the feature order expected by the trained model.

#### `requirements.txt`

Contains the Python dependencies required to run the application.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/RathodChetan1122/AI-Crop-Recommendation-System.git
cd AI-Crop-Recommendation-System
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

Start Streamlit:

```bash
streamlit run app.py
```

The application will normally open in your browser.

You can also access it through the local Streamlit URL shown in the terminal.

---

## 🧾 Example Input

An example input based on the dataset is:

```text
Nitrogen:    90
Phosphorus:  42
Potassium:   43
Temperature: 20.88 °C
Humidity:    82 %
Soil pH:     6.50
Rainfall:    202.94 mm
```

The model can then return a crop recommendation based on the trained model.

---

## 📊 Top-3 Recommendations

The application can display the model's three highest predicted classes, for example:

```text
🥇 Rice
🥈 Jute
🥉 Pomegranate
```

The displayed probabilities are the Random Forest model's predicted class probabilities.

> **Important:** Model probability is not a guarantee that a crop will succeed in the real world.

---

## ⚠️ Farmer-Friendly Decision Support

This application should be treated as a **decision-support system**, not as a replacement for agricultural expertise.

Before cultivation, users should also consider:

- Local weather
- Seasonal conditions
- Irrigation availability
- Water availability
- Soil testing
- Local agricultural practices
- Market demand
- Crop prices
- Pest and disease conditions
- Advice from qualified agricultural experts

---

## 🔐 Model Input Safety

The application preserves the feature order expected by the model:

```text
[
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]
```

The input DataFrame is reordered using the saved feature names before prediction. This helps prevent incorrect predictions caused by supplying features in the wrong order.

---

## 🚀 Deployment

The application is designed to be deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Push the project to GitHub.
2. Make sure `app.py` is in the repository.
3. Make sure the `.pkl` model files are available to the application.
4. Make sure `requirements.txt` contains all required dependencies.
5. Connect the GitHub repository to Streamlit Community Cloud.
6. Select `app.py` as the main application file.
7. Deploy.

The resulting application can be accessed through a public Streamlit URL.

---

## 📦 Requirements

A basic `requirements.txt` can contain:

```text
streamlit
pandas
scikit-learn
joblib
```

For reproducible deployment, package versions can also be pinned to the versions used during model development.

---

## 🔮 Future Enhancements

The project can be extended with additional features such as:

- 🌦️ Live weather API integration
- 📍 Location-based recommendations
- 🌧️ Weather forecast integration
- 💧 Irrigation recommendations
- 🧪 Additional soil properties
- 🌱 Soil nutrient analysis
- 💰 Market-price information
- 📈 Crop profitability estimation
- 🐛 Pest and disease detection
- 📷 Image-based crop/leaf analysis
- 🗣️ Regional-language support
- 🇮🇳 Telugu and other Indian-language interfaces
- 🤖 Agricultural chatbot
- 📱 Mobile-friendly/PWA version
- 👨‍🌾 Farmer profiles
- 📊 Historical prediction dashboard
- 🔔 Weather and farming alerts
- 🧠 More advanced ML models
- ☁️ Cloud database integration

---

## ⚠️ Disclaimer

This application provides a machine-learning-based crop recommendation using the input variables and training data available to the model.

It should **not** be treated as a guaranteed agricultural recommendation.

Actual crop suitability can depend on additional factors that are not included in the current model, including local climate, soil properties, irrigation, season, pests, diseases, market conditions, and farming practices.

Always validate the recommendation with appropriate local agricultural information and professional advice before making cultivation decisions.

---

## 🎓 Academic Project Summary

### Project Title

**AI-Based Crop Recommendation System for Farmers**

### Technology Stack

```text
Python
Pandas
Scikit-learn
Random Forest
Joblib
Streamlit
```

### Dataset

```text
2,200 samples
22 crop classes
7 input features
```

### Final Model

```text
Tuned Random Forest
```

### Final Test Accuracy

```text
99.5455%
```

### Test Performance

```text
438 correct predictions
2 incorrect predictions
440 total test samples
```

---

## 👩‍💻 Project Workflow

```text
Problem Definition
       ↓
Dataset Collection
       ↓
Data Understanding
       ↓
Data Quality Checks
       ↓
Exploratory Data Analysis
       ↓
Feature Analysis
       ↓
Correlation Analysis
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Feature Importance
       ↓
Model Comparison
       ↓
Hyperparameter Tuning
       ↓
Final Model Selection
       ↓
Model Saving
       ↓
Streamlit Application
       ↓
UI Enhancement
       ↓
Deployment
```

---

## ⭐ Project Status

**Status: Completed and deployment-ready**

The machine learning pipeline and Streamlit application are working, with the project prepared for deployment and future feature expansion.

---

## 📌 Future Development Goal

The long-term goal is to evolve this project from a basic crop classification application into a broader **AI-powered agricultural decision-support platform** that combines soil information, weather, location, irrigation, market information, and other agricultural data.
