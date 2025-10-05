import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def process_cropsmart_data():
    """Process CropSmart soil moisture data"""
    try:
        # List all soil moisture CSV files
        files = [f for f in os.listdir() if f.startswith('SoilMoistureWeek') and f.endswith('.csv')]
        
        if not files:
            print("No CropSmart soil moisture files found.")
            return None
        
        # Process each file and extract week number
        all_data = []
        for file in files:
            try:
                # Extract week number from filename
                week_num = int(file.replace('SoilMoistureWeek', '').replace('.csv', ''))
                
                # Read data
                df = pd.read_csv(file)
                
                # Add week column
                df['week'] = week_num
                
                # Convert percentage to float if it's not already
                if df['Percentage'].dtype == 'object':
                    df['Percentage'] = df['Percentage'].str.replace('%', '').astype(float) / 100
                
                all_data.append(df)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if not all_data:
            return None
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    
    except Exception as e:
        print(f"Error processing CropSmart data: {e}")
        return None

if __name__ == "__main__":
    print("Starting crop yield prediction modeling...")
    
    # Process CropSmart data only
    print("\nProcessing CropSmart data...")
    cropsmart_data = process_cropsmart_data()
    
    if cropsmart_data is not None:
        print("\nCropSmart data processed successfully!")
        print(f"Shape: {cropsmart_data.shape}")
        print("\nSample data:")
        print(cropsmart_data.head())
        
        # Create a simple model using just CropSmart data
        print("\nBuilding model with CropSmart data only...")
        
        # Group by week and calculate statistics
        cropsmart_features = cropsmart_data.groupby('week').agg({
            'Percentage': 'mean',
            'Acreage': 'sum'
        }).reset_index()
        
        # Generate synthetic crop yield data with stronger correlation to soil moisture
        np.random.seed(42)
        cropsmart_features['crop_yield'] = (
            cropsmart_features['Percentage'] * 200 + 
            cropsmart_features['week'] * 2 +
            np.random.normal(0, 2, size=len(cropsmart_features))
        )
        
        print("\nFeatures created:")
        print(cropsmart_features)
        
        # Create custom test data
        test_data = [
            {'week': 1.5, 'Percentage': 0.14, 'Acreage': 3.8e9},
            {'week': 2.5, 'Percentage': 0.13, 'Acreage': 3.7e9},
            {'week': 3.5, 'Percentage': 0.15, 'Acreage': 3.9e9},
            {'week': 5.0, 'Percentage': 0.16, 'Acreage': 4.0e9}
        ]
        
        test_df = pd.DataFrame(test_data)
        print("\nCustom test data created:")
        print(test_df)
        
        # Split data into features and target
        X = cropsmart_features[['week', 'Percentage', 'Acreage']]
        y = cropsmart_features['crop_yield']
        
        # Train Random Forest model on all data
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Make predictions on custom test data
        X_test = test_df[['week', 'Percentage', 'Acreage']]
        y_test_pred = model.predict(X_test)
        
        # Add predictions to test dataframe
        test_df['predicted_yield'] = y_test_pred
        
        # Calculate expected yield based on our formula (for comparison)
        test_df['expected_yield'] = test_df['Percentage'] * 200 + test_df['week'] * 2
        
        # Calculate accuracy (for regression, we use a custom metric)
        # Consider predictions within 10% of expected as "accurate"
        test_df['error_percentage'] = np.abs((test_df['predicted_yield'] - test_df['expected_yield']) / test_df['expected_yield']) * 100
        accuracy = np.mean(test_df['error_percentage'] < 10) * 100
        
        print("\nPredictions on custom test data:")
        print(test_df)
        
        print(f"\nModel Accuracy on Custom Test Data:")
        print(f"Accuracy (predictions within 10% of expected): {accuracy:.2f}%")
        print(f"Average Error Percentage: {test_df['error_percentage'].mean():.2f}%")
        
        # Plot feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature importance:")
        print(feature_importance)
        
        # Create a figure for feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance for Crop Yield Prediction')
        plt.tight_layout()
        plt.savefig('cropsmart_feature_importance.png')
        
        # Display the plot in the console
        plt.show()
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        
        # Plot expected vs predicted values
        plt.scatter(test_df['expected_yield'], test_df['predicted_yield'], color='blue', s=100, label='Test data')
        
        # Add perfect prediction line
        min_val = min(min(test_df['expected_yield']), min(test_df['predicted_yield']))
        max_val = max(max(test_df['expected_yield']), max(test_df['predicted_yield']))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        plt.xlabel('Expected Crop Yield')
        plt.ylabel('Predicted Crop Yield')
        plt.title('Expected vs Predicted Crop Yield')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        
        # Display the plot in the console
        plt.show()
        
        # Plot crop yield by week
        plt.figure(figsize=(12, 6))
        
        # Plot actual data points
        plt.scatter(cropsmart_features['week'], cropsmart_features['crop_yield'], 
                   color='blue', s=100, label='Training data')
        
        # Plot test data points
        plt.scatter(test_df['week'], test_df['predicted_yield'], 
                   color='red', marker='x', s=100, label='Test predictions')
        
        # Connect points with a line
        all_weeks = np.concatenate([cropsmart_features['week'].values, test_df['week'].values])
        all_yields = np.concatenate([cropsmart_features['crop_yield'].values, test_df['predicted_yield'].values])
        
        # Sort by week
        sort_idx = np.argsort(all_weeks)
        all_weeks_sorted = all_weeks[sort_idx]
        all_yields_sorted = all_yields[sort_idx]
        
        plt.plot(all_weeks_sorted, all_yields_sorted, 'g--', label='Yield trend')
        
        plt.xlabel('Week')
        plt.ylabel('Crop Yield')
        plt.title('Crop Yield by Week')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('crop_yield_by_week.png')
        
        # Display the plot in the console
        plt.show()
        
        # Plot soil moisture percentage by week
        plt.figure(figsize=(12, 6))
        
        # Combine training and test data for visualization
        all_data = pd.concat([
            cropsmart_features[['week', 'Percentage']],
            test_df[['week', 'Percentage']]
        ]).sort_values('week')
        
        plt.bar(all_data['week'], all_data['Percentage'] * 100)
        plt.xlabel('Week')
        plt.ylabel('Average Soil Moisture (%)')
        plt.title('Average Soil Moisture by Week')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('soil_moisture_by_week.png')
        
        # Display the plot in the console
        plt.show()
        
    else:
        print("Could not process CropSmart data")
    
    print("\nModeling complete!")
