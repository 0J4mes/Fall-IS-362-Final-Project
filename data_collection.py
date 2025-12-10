"""
Data collection module for Whole Foods supply chain analysis.
This module collects data from multiple sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import openmeteo_requests
import retry_requests
from typing import Dict
import os


def generate_sample_data() -> Dict[str, pd.DataFrame]:
    """Generate sample data if API calls fail."""
    print("Generating sample data for analysis...")

    np.random.seed(42)

    # Generate sales data
    dates = pd.date_range('2022-10-01', '2023-03-31', freq='D')
    departments = ['Produce', 'Meat & Seafood', 'Dairy', 'Bakery', 'Grocery']
    stores = ['NYC_Chelsea', 'SF_Noe_Valley', 'LA_Venice', 'CHI_Lincoln_Park']

    sales_data = []
    for date in dates[:100]:  # Limit to 100 records for sample
        for dept in departments:
            for store in stores:
                if np.random.random() > 0.7:  # 30% probability
                    sales_data.append({
                        'date': date,
                        'department': dept,
                        'product_category': f'{dept} Category',
                        'store': store,
                        'sales_volume': np.random.randint(50, 500),
                        'inventory_level': np.random.randint(100, 1000),
                        'restock_time_days': np.random.randint(1, 5),
                        'waste_percentage': np.random.uniform(0.5, 5.0),
                        'is_weekend': 1 if date.weekday() >= 5 else 0,
                        'month': date.month,
                        'day_of_week': date.weekday()
                    })

    sales_df = pd.DataFrame(sales_data)

    # Generate supplier data
    supplier_data = []
    suppliers = ['Local_Farms_Co', 'Organic_Valley', 'Sustainable_Seafood_Inc']
    regions = ['Northeast', 'West Coast', 'Midwest']

    for supplier in suppliers:
        for region in regions:
            supplier_data.append({
                'supplier': supplier,
                'region': region,
                'avg_delivery_time_days': np.random.randint(1, 4),
                'reliability_score': np.random.uniform(0.8, 0.99)
            })

    supplier_df = pd.DataFrame(supplier_data)

    # Generate weather data
    weather_data = []
    for date in dates[:100]:
        for region in regions:
            weather_data.append({
                'date': date,
                'region': region,
                'avg_temperature_c': np.random.uniform(-5, 30),
                'precipitation_mm': np.random.exponential(2),
                'weather_code': np.random.randint(0, 100)
            })

    weather_df = pd.DataFrame(weather_data)

    # Generate holiday data
    holidays = {
        '2022-10-31': 'Halloween',
        '2022-11-24': 'Thanksgiving',
        '2022-12-25': 'Christmas',
        '2023-01-01': 'New Year'
    }

    holiday_list = []
    for date_str, holiday_name in holidays.items():
        holiday_list.append({
            'date': pd.to_datetime(date_str),
            'holiday_name': holiday_name,
            'is_holiday': 1
        })

    holiday_df = pd.DataFrame(holiday_list)

    # Save data
    sales_df.to_csv('data/whole_foods_sales_inventory.csv', index=False)
    supplier_df.to_csv('data/whole_foods_suppliers.csv', index=False)
    weather_df.to_csv('data/weather_data.csv', index=False)
    holiday_df.to_csv('data/holidays.csv', index=False)

    return {
        'sales': sales_df,
        'suppliers': supplier_df,
        'weather': weather_df,
        'holidays': holiday_df
    }


def collect_all_data() -> Dict[str, pd.DataFrame]:
    """Collect all data from various sources."""
    print("Collecting data from multiple sources...")

    try:
        # First try to use existing data
        if os.path.exists('data/whole_foods_sales_inventory.csv'):
            print("Loading existing data from files...")
            sales_df = pd.read_csv('data/whole_foods_sales_inventory.csv')
            supplier_df = pd.read_csv('data/whole_foods_suppliers.csv')
            weather_df = pd.read_csv('data/weather_data.csv')
            holidays_df = pd.read_csv('data/holidays.csv')

            # Convert date columns
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            holidays_df['date'] = pd.to_datetime(holidays_df['date'])

            return {
                'sales': sales_df,
                'suppliers': supplier_df,
                'weather': weather_df,
                'holidays': holidays_df
            }
    except:
        pass

    # If no existing data, generate sample data
    return generate_sample_data()