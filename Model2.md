# 🌱 Environmental Data Analysis and Crop Suitability Prediction

This project uses **time-series environmental data** (Precipitation, Soil Moisture, and Temperatures) from **2018 to 2020** to build a **classification model** that predicts the **daily suitability of the environment for general crop growth**.

---

## 📊 1. Data Sources and Ingestion

Four distinct daily time-series datasets were loaded and merged:

- `Precipitation.csv`  
- `SoilMoistures.csv`  
- `SoilTemperature.csv`  
- `SurfaceTemp.csv`  

The data was cleaned by skipping the initial **8 metadata rows** and merging the resulting DataFrames on their common **Date index**.

| Variable Name | Unit | Source File |
|----------------|------|--------------|
| Precipitation_Rate_mm_hr | mm/hr | Precipitation.csv |
| Soil_Moisture_m3_m3 | m³/m³ | SoilMoistures.csv |
| Soil_Temperature_K | K | SoilTemperature.csv |
| Surface_Temperature_K | K | SurfaceTemp.csv |

---

## 🌾 2. Target Variable Definition — *Crop Growth Suitability*

Since the original data did not contain a target variable for crop yield or growth, a **proxy binary target**, `Crop_Growth_Suitability`, was engineered based on generalized agricultural principles.

A day was classified as **Suitable (1)** if **all three** conditions below were met; otherwise, it was classified as **Not Suitable (0)**.

| Condition | Threshold | Rationale |
|------------|------------|------------|
| **Soil Moisture** | > 0.10 m³/m³ | Ensures adequate water availability, avoiding drought stress. |
| **Soil Temperature** | > 15°C (≈288.15 K) | Minimum temperature for active root growth and nutrient uptake. |
| **Surface Temperature** | > 15°C (≈288.15 K) | Minimum temperature for above-ground plant metabolism and growth. |

---

## ⚙️ 3. Feature Engineering

To capture **time-series dynamics** and **seasonality**, several features were added to the dataset.

### ⏳ Time-Based Features
- `Month`
- `Dayofyear`

### 🔁 Lagged Features
The **previous day’s values (Lag = 1)** for each primary environmental feature were included, allowing the model to predict today’s suitability using yesterday’s conditions:
- `Precipitation_Rate_mm_hr_lag1`
- `Soil_Moisture_m3_m3_lag1`
- `Soil_Temperature_K_lag1`
- `Surface_Temperature_K_lag1`

---

## 🤖 4. Machine Learning Model and Methodology

### 🧠 Model Used
| Parameter | Description |
|------------|-------------|
| **Model** | Random Forest Classifier |
| **Purpose** | Classification (predicting 0 or 1) |
| **Parameters** | `n_estimators=200`, `max_depth=10`, `class_weight='balanced'` |

### 🧪 Evaluation Methodology
| Aspect | Description |
|---------|--------------|
| **Data Split** | Chronological — first 80% for training, last 20% for testing |
| **Reasoning** | Ensures the model is evaluated on *future (unseen)* data — essential for time-series validation |
| **Metrics Used** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Result Interpretation** | The model achieved high accuracy in predicting the proxy target. Feature Importance analysis revealed that **Surface Temperature** and **Soil Temperature** were the most influential factors. |

---

## 🧩 5. Hypothetical Scenario Testing

To demonstrate the model's interpretability and decision-making process, **three custom fictional scenarios** were tested.

### 🧱 Scenario Setup
- **Base Conditions:** Average "yesterday" conditions and neutral time features (e.g., `Month=6`, `Dayofyear=160`).
- **Goal:** Observe model response to different environmental combinations.

| Scenario Name | Key Condition Tested (Today's Values) | Expected Outcome | Explanation |
|----------------|--------------------------------------|------------------|--------------|
| **Ideal Growth** | High temperatures (>30°C), High moisture (~0.35 m³/m³) | Suitable (1) | Meets all thresholds |
| **Marginal Moisture** | Warm temps (~21°C), Low moisture (~0.09 m³/m³) | Not Suitable (0) | Fails moisture threshold |
| **Cold Snap** | Low temps (~5°C), Adequate moisture (~0.18 m³/m³) | Not Suitable (0) | Fails temperature thresholds |

---

## 📈 6. Visualization of Scenario Predictions

The model’s output probabilities were visualized in a **bar chart**, showing the predicted **likelihood of suitability** for each scenario.

✅ **Result:**  
The visualization confirmed that the model correctly follows the agricultural logic — assigning:
- High suitability to **Ideal Growth**
- Low suitability to **Marginal Moisture** and **Cold Snap**

---

## 🧠 Summary

This project successfully:
- Integrated multiple environmental time-series datasets  
- Engineered biologically meaningful features  
- Built and evaluated a **Random Forest classification model**  
- Demonstrated logical interpretability through custom scenario testing  

---

## 🧰 Technologies Used

| Library | Purpose |
|----------|----------|
| `pandas`, `numpy` | Data manipulation and computation |
| `matplotlib`, `seaborn` | Data visualization |
| `scikit-learn` | Model training and evaluation |

---

