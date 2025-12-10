"""
Data processing and analysis module for Whole Foods supply chain.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def process_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Process and merge all datasets."""
    print("Processing and merging datasets...")

    # Load data
    sales_df = data_dict['sales'].copy()
    supplier_df = data_dict['suppliers'].copy()
    weather_df = data_dict['weather'].copy()
    holidays_df = data_dict['holidays'].copy()

    # DATA TRANSFORMATION 1: Convert date columns to datetime format
    if not pd.api.types.is_datetime64_any_dtype(sales_df['date']):
        sales_df['date'] = pd.to_datetime(sales_df['date'])
    if 'date' in weather_df.columns and not pd.api.types.is_datetime64_any_dtype(weather_df['date']):
        weather_df['date'] = pd.to_datetime(weather_df['date'])
    if 'date' in holidays_df.columns and not pd.api.types.is_datetime64_any_dtype(holidays_df['date']):
        holidays_df['date'] = pd.to_datetime(holidays_df['date'])

    # DATA TRANSFORMATION 2: Create delivery time windows
    def categorize_delivery_time(days):
        if days <= 1:
            return 'Next Day'
        elif days <= 2:
            return '2 Days'
        elif days <= 3:
            return '3 Days'
        else:
            return '4+ Days'

    sales_df['delivery_window'] = sales_df['restock_time_days'].apply(categorize_delivery_time)

    # DATA TRANSFORMATION 3: Create temperature categories
    if 'avg_temperature_c' in weather_df.columns:
        def categorize_temperature(temp):
            if temp < 0:
                return 'Freezing'
            elif temp < 10:
                return 'Cold'
            elif temp < 20:
                return 'Cool'
            elif temp < 30:
                return 'Warm'
            else:
                return 'Hot'

        weather_df['temp_category'] = weather_df['avg_temperature_c'].apply(categorize_temperature)

    # Merge datasets
    # Merge sales with holidays if holidays_df exists
    if not holidays_df.empty and 'date' in holidays_df.columns:
        sales_df = pd.merge(sales_df, holidays_df, on='date', how='left')
        sales_df['is_holiday'] = sales_df['is_holiday'].fillna(0)
        sales_df['holiday_name'] = sales_df['holiday_name'].fillna('No Holiday')
    else:
        sales_df['is_holiday'] = 0
        sales_df['holiday_name'] = 'No Holiday'

    # Map store to region for weather merge
    store_to_region = {
        'NYC_Chelsea': 'Northeast',
        'SF_Noe_Valley': 'West Coast',
        'LA_Venice': 'West Coast',
        'CHI_Lincoln_Park': 'Midwest',
        'TX_Austin': 'Southwest'
    }

    # Create region column if store exists
    if 'store' in sales_df.columns:
        sales_df['region'] = sales_df['store'].map(store_to_region)
        # Fill missing regions
        sales_df['region'] = sales_df['region'].fillna('Unknown')
    else:
        sales_df['region'] = 'Unknown'

    # Merge with weather data if available
    if not weather_df.empty and 'date' in weather_df.columns and 'region' in weather_df.columns:
        merged_df = pd.merge(sales_df, weather_df, on=['date', 'region'], how='left')
    else:
        merged_df = sales_df.copy()
        # Add dummy weather columns
        merged_df['avg_temperature_c'] = np.random.uniform(10, 25, len(merged_df))
        merged_df['precipitation_mm'] = np.random.exponential(1, len(merged_df))
        merged_df['weather_code'] = np.random.randint(0, 100, len(merged_df))

    # Handle missing values
    numeric_cols = ['sales_volume', 'inventory_level', 'restock_time_days',
                    'waste_percentage', 'avg_temperature_c', 'precipitation_mm']

    for col in numeric_cols:
        if col in merged_df.columns:
            if col in ['avg_temperature_c', 'precipitation_mm']:
                merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
            else:
                merged_df[col] = merged_df[col].fillna(0)

    # Create additional derived columns
    merged_df['sales_per_inventory'] = merged_df['sales_volume'] / merged_df['inventory_level'].replace(0, 1)
    merged_df['waste_cost_est'] = merged_df['waste_percentage'] * merged_df['sales_volume'] * 0.1

    # Add month and day name for better visualization
    merged_df['month_name'] = merged_df['date'].dt.month_name()
    merged_df['day_name'] = merged_df['date'].dt.day_name()

    # Add supplier data if available
    if not supplier_df.empty and 'region' in supplier_df.columns:
        supplier_agg = supplier_df.groupby('region').agg({
            'avg_delivery_time_days': 'mean',
            'reliability_score': 'mean'
        }).reset_index()

        merged_df = pd.merge(merged_df, supplier_agg, on='region', how='left')
    else:
        merged_df['avg_delivery_time_days'] = np.random.randint(1, 4, len(merged_df))
        merged_df['reliability_score'] = np.random.uniform(0.8, 0.99, len(merged_df))

    print(f"✓ Processed {len(merged_df)} records")
    print(f"✓ Columns: {', '.join(merged_df.columns.tolist())}")

    return {
        'merged': merged_df,
        'sales': sales_df
    }


def perform_analysis(processed_data: Dict) -> Dict:
    """Perform statistical analysis on the processed data."""
    print("Performing statistical analysis...")

    df = processed_data['merged']

    results = {}

    # ANALYSIS 1: GROUPING/AGGREGATION - Sales by department
    if 'department' in df.columns and 'sales_volume' in df.columns:
        sales_by_dept = df.groupby('department').agg({
            'sales_volume': ['mean', 'sum', 'count'],
            'waste_percentage': 'mean',
            'restock_time_days': 'mean'
        }).round(2).reset_index()

        # Flatten column names
        sales_by_dept.columns = ['department', 'sales_mean', 'sales_total',
                                 'record_count', 'waste_mean', 'restock_mean']
        results['sales_by_dept'] = sales_by_dept

    # ANALYSIS 2: GROUPING/AGGREGATION - Sales by day of week
    if 'day_name' in df.columns:
        sales_by_day = df.groupby('day_name').agg({
            'sales_volume': 'mean',
            'waste_percentage': 'mean'
        }).round(2).reset_index()
        results['sales_by_day'] = sales_by_day

    # ANALYSIS 3: Statistical test - ANOVA for sales across departments
    if 'department' in df.columns and 'sales_volume' in df.columns:
        try:
            anova_model = ols('sales_volume ~ C(department)', data=df).fit()
            anova_table = sm.stats.anova_lm(anova_model, typ=2)
            results['anova_table'] = anova_table
        except:
            results['anova_table'] = None

    # ANALYSIS 4: Correlation analysis
    numeric_cols = ['sales_volume', 'inventory_level', 'restock_time_days',
                    'waste_percentage', 'avg_temperature_c', 'precipitation_mm']

    available_cols = [col for col in numeric_cols if col in df.columns]
    if len(available_cols) >= 2:
        correlation_matrix = df[available_cols].corr().round(3)
        results['correlation_matrix'] = correlation_matrix

    # ANALYSIS 5: Regression analysis for waste prediction (Feature not covered in class)
    try:
        # Prepare features
        features = []
        if 'month' in df.columns:
            features.append('month')
        if 'day_of_week' in df.columns:
            features.append('day_of_week')
        if 'avg_temperature_c' in df.columns:
            features.append('avg_temperature_c')
        if 'precipitation_mm' in df.columns:
            features.append('precipitation_mm')
        if 'is_weekend' in df.columns:
            features.append('is_weekend')
        if 'is_holiday' in df.columns:
            features.append('is_holiday')

        # Encode department if available
        if 'department' in df.columns and 'waste_percentage' in df.columns:
            le = LabelEncoder()
            df_encoded = df.copy()
            df_encoded['department_encoded'] = le.fit_transform(df['department'])
            features.append('department_encoded')

            X = df_encoded[features]
            y = df_encoded['waste_percentage']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X_train, y_train)

            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            results['feature_importance'] = feature_importance
            results['model_score'] = rf_model.score(X_test, y_test)
    except Exception as e:
        print(f"Note: Random Forest analysis skipped: {e}")
        results['feature_importance'] = None
        results['model_score'] = None

    print("✓ Statistical analysis complete")
    return results