# 🌱 Environmental Data Analysis and Crop Suitability Prediction

This project leverages **time-series environmental data (Precipitation, Soil Moisture, and Temperature)** from **2018–2020** to predict the **daily suitability of environmental conditions** for general crop growth using **machine learning**.

---

## 📊 1. Data Sources and Ingestion

Four distinct daily time-series datasets were combined and cleaned to form the analysis base.

| Dataset | Description | Unit | Source File |
|----------|--------------|------|--------------|
| Precipitation_Rate_mm_hr | Daily precipitation rate | mm/hr | `Precipitation.csv` |
| Soil_Moisture_m3_m3 | Soil moisture content | m³/m³ | `SoilMoistures.csv` |
| Soil_Temperature_K | Soil temperature | Kelvin (K) | `SoilTemperature.csv` |
| Surface_Temperature_K | Surface temperature | Kelvin (K) | `SurfaceTemp.csv` |

### 🧹 Data Cleaning
- Skipped initial **8 metadata rows** in each CSV file.  
- Parsed the **Date** column and set it as the index.  
- Merged all datasets into a single DataFrame on their common **Date index**.

---

## 🌾 2. Target Variable Definition — *Crop Growth Suitability*

Since the original dataset did not include crop yield or growth labels, a **proxy binary target variable** (`Crop_Growth_Suitability`) was engineered using agricultural thresholds.

| Condition | Threshold | Rationale |
|------------|------------|------------|
| **Soil Moisture** | > 0.10 m³/m³ | Ensures adequate water availability, avoiding drought stress. |
| **Soil Temperature** | > 15°C | Represents the minimum temperature required for active root growth and nutrient uptake. |
| **Surface Temperature** | > 15°C | Represents the minimum temperature for above-ground plant metabolism and growth. |

🔹 If **all three** conditions were satisfied → `Crop_Growth_Suitability = 1 (Suitable)`  
🔹 Otherwise → `Crop_Growth_Suitability = 0 (Not Suitable)`

---

## ⚙️ 3. Feature Engineering

To help the model learn temporal trends and seasonal effects, several additional features were generated.

### 🕒 Time-Based Features
- **Month** — Extracted from the date to capture seasonal variations.
- **DayOfYear** — Captures cyclical changes within the year.

### 🔁 Lagged Features
- Added **previous day (lag-1)** values of key environmental features:
  - `Precipitation_Rate_mm_hr_lag1`
  - `Soil_Moisture_m3_m3_lag1`
  - `Soil_Temperature_K_lag1`
  - `Surface_Temperature_K_lag1`

This allows the model to use yesterday’s environmental conditions to predict today’s crop suitability.

---

## 🤖 4. Machine Learning Model and Methodology

### 🧠 Model Used
| Parameter | Description |
|------------|-------------|
| **Algorithm** | Random Forest Classifier |
| **Task Type** | Binary Classification (Suitable = 1, Not Suitable = 0) |
| **Purpose** | Predicting daily environmental suitability for crop growth |

### 🧪 Model Parameters
- `n_estimators = 100`  
- `random_state = 42`  
- `max_depth = None` (allowing trees to expand fully for higher accuracy)

---

## 📈 5. Evaluation Methodology

| Aspect | Description |
|---------|--------------|
| **Data Split** | 80% training (earlier data) / 20% testing (later data) |
| **Validation Type** | Chronological split to simulate future prediction |
| **Metrics Used** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Interpretation** | Model performance was robust, showing strong accuracy and consistency across metrics. |

### 🔍 Insights
- **Surface Temperature** and **Soil Temperature** emerged as the **most important predictors**.  
- The model effectively captures **daily environmental variability** influencing crop growth.

---

## 📊 6. Visualizations (Optional)
Depending on your implementation, the following plots can be generated:
- **Feature Importance Plot** — shows which environmental factors influence suitability most.  
- **Confusion Matrix Heatmap** — visualizes model accuracy and false predictions.  
- **Time-Series Plots** — track environmental variable trends across seasons.

---

## 🧩 7. Libraries Used
| Library | Purpose |
|----------|----------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Machine learning model and evaluation |

---

## 🧠 Summary

**Workflow Overview:**
