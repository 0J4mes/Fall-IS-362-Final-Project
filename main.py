#!/usr/bin/env python3
"""
IS 362 Final Project: Whole Foods Supply Chain Analysis
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import local modules directly
from data_collection import collect_all_data
from data_processing import process_data, perform_analysis
from visualization import create_visualizations, create_interactive_dashboard


def main():
    """Main execution function for the Whole Foods supply chain analysis."""

    print("=" * 60)
    print("WHOLE FOODS SUPPLY CHAIN ANALYSIS")
    print("IS 362 Final Project")
    print("=" * 60)

    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Step 1: Collect data from multiple sources
    print("\n[1/4] Collecting data from multiple sources...")
    try:
        data_dict = collect_all_data()
        print("✓ Data collection complete")
    except Exception as e:
        print(f"✗ Error in data collection: {e}")
        print("Continuing with sample data...")
        # Generate minimal sample data if collection fails
        from data_collection import generate_sample_data
        data_dict = generate_sample_data()

    # Step 2: Process and clean data
    print("\n[2/4] Processing and cleaning data...")
    try:
        processed_data = process_data(data_dict)
        print("✓ Data processing complete")
    except Exception as e:
        print(f"✗ Error in data processing: {e}")
        return

    # Step 3: Perform statistical analysis
    print("\n[3/4] Performing statistical analysis...")
    try:
        analysis_results = perform_analysis(processed_data)
        print("✓ Statistical analysis complete")
    except Exception as e:
        print(f"✗ Error in statistical analysis: {e}")
        return

    # Step 4: Create visualizations
    print("\n[4/4] Creating visualizations and dashboard...")
    try:
        create_visualizations(processed_data, analysis_results)
        create_interactive_dashboard(processed_data)
        print("✓ Visualizations created")
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        return

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("✓ Check the 'outputs/' folder for visualizations")
    print("✓ Interactive dashboard saved as 'dashboard.html'")
    print("✓ Jupyter notebook available at 'whole_foods_analysis.ipynb'")
    print("=" * 60)

    # Show summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    df = processed_data['merged']
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Departments analyzed: {len(df['department'].unique())}")
    print(f"Stores analyzed: {len(df['store'].unique())}")
    print(f"Average sales volume: {df['sales_volume'].mean():.0f}")
    print(f"Average waste percentage: {df['waste_percentage'].mean():.2f}%")


if __name__ == "__main__":
    main()