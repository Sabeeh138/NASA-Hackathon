import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 1. Data Ingestion and Merging (Replicating and expanding initial steps) ---

def load_and_merge_data():
    """
    Loads, cleans, and merges the four time-series environmental data files.
    """
    print("Loading and merging environmental data...")
    file_names = {
        'Precipitation.csv': 'Precipitation_Rate_mm_hr',
        'SoilMoistures.csv': 'Soil_Moisture_m3_m3',
        'SoilTemperature.csv': 'Soil_Temperature_K',
        'SurfaceTemp.csv': 'Surface_Temperature_K'
    }

    dataframes = {}

    for file_name, new_col_name in file_names.items():
        # Load, skipping initial metadata rows
        try:
            df = pd.read_csv(file_name, skiprows=8)
        except FileNotFoundError:
            print(f"Error: {file_name} not found.")
            continue

        # Clean up column names
        df.columns = [col.strip() for col in df.columns]

        # Find the time and data column names (assuming they start with 'time' and 'mean'/'precip'/'radt')
        time_col = next((col for col in df.columns if col.lower().startswith('time')), None)
        data_col = next((col for col in df.columns if col.lower().startswith(('mean', 'precip', 'soil', 'radt'))), None)

        if time_col and data_col:
            # Select and rename columns
            df = df[[time_col, data_col]]
            df.rename(columns={time_col: 'Date', data_col: new_col_name}, inplace=True)

            # Convert 'Date' to datetime and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            dataframes[file_name] = df
        else:
            print(f"Warning: Could not identify time or data columns in {file_name}. Skipping.")


    # Merge the dataframes based on their common time index
    merged_df = pd.concat(dataframes.values(), axis=1)

    print(f"Data merged successfully. Total days: {len(merged_df)}")
    return merged_df

# --- 2. Feature Engineering and Target Variable Creation ---

def prepare_data(df):
    """
    Creates the target variable, adds time-series and lagged features, and cleans the final dataset.
    """
    print("Creating target variable and engineering features...")

    # --- Target Variable: Crop Growth Suitability (Based on General Agricultural Rules) ---
    # Convert temperatures from Kelvin (K) to Celsius (°C) for easier interpretation (288.15 K = 15°C)
    df['Soil_Temperature_C'] = df['Soil_Temperature_K'] - 273.15
    df['Surface_Temperature_C'] = df['Surface_Temperature_K'] - 273.15

    # Defining suitability conditions (adjust these based on target crop if known)
    TEMP_THRESHOLD_C = 15.0  # 15°C is a common active growth threshold
    MOISTURE_THRESHOLD = 0.10 # 0.10 m^3/m^3 is a proxy for minimum required soil moisture

    soil_moisture_ok = df['Soil_Moisture_m3_m3'] > MOISTURE_THRESHOLD
    soil_temp_ok = df['Soil_Temperature_C'] > TEMP_THRESHOLD_C
    surface_temp_ok = df['Surface_Temperature_C'] > TEMP_THRESHOLD_C

    # Target is 1 if ALL conditions are met, 0 otherwise
    df['Crop_Growth_Suitability'] = (soil_moisture_ok & soil_temp_ok & surface_temp_ok).astype(int)

    # --- Time-Based Features (Seasonality) ---
    df['Month'] = df.index.month
    df['Dayofyear'] = df.index.dayofyear

    # --- Lagged Features (Time-Series Prediction) ---
    # Use yesterday's values (lag=1) to predict today's suitability
    lag_days = 1
    for col in ['Precipitation_Rate_mm_hr', 'Soil_Moisture_m3_m3', 'Soil_Temperature_K', 'Surface_Temperature_K']:
        df[f'{col}_Lag_{lag_days}'] = df[col].shift(lag_days)

    # Drop the first row which now contains NaN due to lagging
    df.dropna(inplace=True)

    print(f"Dataset size after feature engineering and cleaning: {len(df)}")
    print(f"Target distribution (1=Suitable, 0=Not Suitable):\n{df['Crop_Growth_Suitability'].value_counts()}")

    return df

# --- 3. Machine Learning Model Training and Evaluation ---

def train_and_evaluate_model(df):
    """
    Splits data, trains a Random Forest Classifier, and reports results.
    Returns the trained model and the feature list.
    """
    print("\n--- Training Machine Learning Model ---")

    # Features to use for the final model (Current time, time-based, and lagged environmental data)
    current_features = ['Soil_Temperature_K', 'Surface_Temperature_K', 'Soil_Moisture_m3_m3', 'Precipitation_Rate_mm_hr']
    lagged_features = [col for col in df.columns if '_Lag_1' in col]
    time_features = ['Month', 'Dayofyear']
    all_features = current_features + lagged_features + time_features

    # We use all features to predict suitability
    X = df[all_features]
    y = df['Crop_Growth_Suitability']

    # Split data chronologically (better for time series than random split)
    # We use the first 80% for training and the last 20% for testing
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=200,      # More trees for robustness
        max_depth=10,          # Limit depth to prevent overfitting
        random_state=42,
        class_weight='balanced' # Useful since the classes are not perfectly balanced
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # --- Evaluation ---
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Suitable (0)', 'Suitable (1)'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("\n--- Model Performance Report ---")
    print(f"Model: Random Forest Classifier")
    print(f"Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix (Rows=Actual, Columns=Predicted):\n", conf_matrix)
    print("\nTop 10 Feature Importance:\n", feature_importance.head(10))

    return model, all_features, df, y_test, y_pred

# --- 4. Visualization of Historical Performance ---

def plot_suitability(df, y_test, y_pred):
    """
    Generates a plot showing the ground truth suitability over time.
    """
    # Create a column for the prediction results on the test set
    df['Predicted_Suitability'] = np.nan
    df.loc[y_test.index, 'Predicted_Suitability'] = y_pred

    # Plotting
    plt.figure(figsize=(14, 6))

    # Plot the full ground truth target
    df['Crop_Growth_Suitability'].plot(
        ax=plt.gca(),
        marker='.',
        linestyle='-',
        label='Actual Suitability (Training & Test)',
        color='grey',
        alpha=0.4
    )

    # Highlight the test set region predictions
    df.loc[y_test.index, 'Predicted_Suitability'].plot(
        ax=plt.gca(),
        marker='o',
        linestyle='None',
        markersize=3,
        label='Predicted Suitability (Test Set)',
        color='blue'
    )

    plt.title('Daily Crop Growth Suitability (1=Suitable, 0=Not Suitable) and Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Suitability Score')
    plt.yticks([0, 1], ['Not Suitable (0)', 'Suitable (1)'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 5. Hypothetical Scenario Prediction and Visualization (New Function) ---

def predict_and_plot_hypothetical_scenarios(model, features):
    """
    Creates hypothetical scenarios, predicts suitability, and visualizes results.
    """
    print("\n--- Predicting Hypothetical Scenarios ---")

    # The model uses today's data (Temp, Moisture) + Yesterday's data (Lag_1) + Time (Month, Dayofyear).
    # For a simple test, we will assume "Yesterday's" conditions (Lag_1) are set to an average value from the dataset,
    # and focus the test on TODAY's conditions.

    # Base values (representing a non-extreme day in June)
    base_values = {
        'Month': 6,
        'Dayofyear': 160, # Early June
        'Precipitation_Rate_mm_hr_Lag_1': 0.05,
        'Soil_Moisture_m3_m3_Lag_1': 0.15,
        'Soil_Temperature_K_Lag_1': 295.0,
        'Surface_Temperature_K_Lag_1': 298.0,
    }

    # Define three scenarios focusing on TODAY's environmental conditions
    scenarios = {
        # Scenario 1: Ideal Conditions (All thresholds easily met)
        'Ideal Growth': {
            'Soil_Temperature_K': 305.0,   # 31.85 C
            'Surface_Temperature_K': 308.0,  # 34.85 C
            'Soil_Moisture_m3_m3': 0.35,
            'Precipitation_Rate_mm_hr': 0.1,
        },
        # Scenario 2: Marginal Conditions (Soil Moisture just below threshold)
        'Marginal Moisture': {
            'Soil_Temperature_K': 295.0,  # 21.85 C
            'Surface_Temperature_K': 298.0,  # 24.85 C
            'Soil_Moisture_m3_m3': 0.09,    # Below 0.10 threshold
            'Precipitation_Rate_mm_hr': 0.02,
        },
        # Scenario 3: Poor Conditions (Temperature too low, e.g., in winter)
        'Cold Snap': {
            'Soil_Temperature_K': 280.0,   # 6.85 C (Below 15 C threshold)
            'Surface_Temperature_K': 275.0,  # 1.85 C (Below 15 C threshold)
            'Soil_Moisture_m3_m3': 0.18,
            'Precipitation_Rate_mm_hr': 0.01,
        }
    }

    # Create the DataFrame for prediction
    test_data = []
    for name, current_data in scenarios.items():
        # Combine base values (lagged/time) with current values
        row = {**base_values, **current_data}
        test_data.append(row)

    X_hypothetical = pd.DataFrame(test_data, index=scenarios.keys())
    # Re-order columns to match the training data feature order
    X_hypothetical = X_hypothetical[features]

    # Predict the suitability
    predictions = model.predict(X_hypothetical)
    probabilities = model.predict_proba(X_hypothetical)
    results_df = X_hypothetical.copy()
    results_df['Predicted_Suitability'] = predictions
    results_df['Prob_Suitable (1)'] = probabilities[:, 1]
    results_df['Prob_Not_Suitable (0)'] = probabilities[:, 0]

    print("\nPrediction Results:")
    print(results_df[['Soil_Temperature_K', 'Soil_Moisture_m3_m3', 'Predicted_Suitability', 'Prob_Suitable (1)']])

    # Visualization
    plt.figure(figsize=(10, 5))
    bar_colors = np.where(predictions == 1, 'seagreen', 'firebrick')
    plt.bar(results_df.index, results_df['Prob_Suitable (1)'], color=bar_colors)

    for i, pred in enumerate(predictions):
        label = 'Suitable (1)' if pred == 1 else 'Not Suitable (0)'
        plt.text(i, results_df['Prob_Suitable (1)'].iloc[i] + 0.03,
                 f'Prediction: {label}', ha='center', color='black', fontsize=10, fontweight='bold')

    plt.ylim(0, 1.1)
    plt.title('Model Prediction for Hypothetical Environmental Scenarios')
    plt.ylabel('Predicted Probability of Suitability (1)')
    plt.xlabel('Scenario')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load and Merge Data
    merged_data = load_and_merge_data()

    if merged_data.empty:
        print("Could not process or merge data. Exiting.")
    else:
        # 2. Feature Engineering and Target Creation
        processed_data = prepare_data(merged_data)

        # 3. Model Training and Evaluation
        model, all_features, final_df, y_test, y_pred = train_and_evaluate_model(processed_data)

        # 4. Visualization of Historical Performance (Original Plot)
        plot_suitability(final_df, y_test, y_pred)

        # 5. Visualization of Hypothetical Scenarios (New Plot)
        predict_and_plot_hypothetical_scenarios(model, all_features)
