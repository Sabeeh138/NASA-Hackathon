## 🌾 CropSmart Soil Moisture & Crop Yield Prediction

This project processes weekly soil moisture data from CropSmart and builds a machine learning model to predict crop yield using soil moisture, acreage, and time (week).

### Features:
- Automatically loads all `SoilMoistureWeekX.csv` files
- Cleans and combines them into one dataset
- Generates synthetic yield data correlated with soil moisture
- Trains a Random Forest Regressor
- Tests predictions on custom input data
- Evaluates accuracy (within 10% error threshold)
- Visualizes:
  - Feature importance
  - Expected vs predicted yield
  - Yield trend over weeks
  - Soil moisture distribution

### Output:
The script generates:
- `cropsmart_feature_importance.png`
- `actual_vs_predicted.png`
- `crop_yield_by_week.png`
- `soil_moisture_by_week.png`

### Libraries Used:
`pandas`, `numpy`, `matplotlib`, `sklearn`
