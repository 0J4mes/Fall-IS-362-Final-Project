# Fall-IS-362-Final-Project
# Whole Foods Supply Chain Analysis

## Project Overview
This project analyzes Whole Foods Market's supply chain efficiency, focusing on inventory management, sales trends, and operational factors affecting performance. The analysis identifies patterns related to seasonality, weather, holidays, and departmental differences to provide actionable insights for improving inventory turnover and reducing waste.

## Project Requirements Met
✅ **Data Sources (4 types):**
1. Simulated Whole Foods sales/inventory data (CSV)
2. Simulated supplier data (CSV)
3. Weather data from Open-Meteo API
4. Web-scraped holiday calendar

✅ **Data Transformations:**
1. Date format conversion
2. Creation of delivery time windows
3. Temperature categorization
4. Wide to long format conversion
5. Derived metrics (sales_per_inventory, waste_cost_est)

✅ **Grouping/Aggregation:**
1. Sales by department and day of week
2. Seasonal trends analysis
3. Store-level statistics

✅ **Statistical Analysis:**
1. ANOVA test for department sales differences
2. Correlation matrix
3. Time series decomposition
4. Random Forest regression (feature not covered in class)

✅ **Visualizations:**
1. Bar charts (department performance)
2. Line charts (seasonal trends)
3. Heatmaps (sales patterns)
4. Box plots (waste distribution)
5. Interactive Folium map (bonus feature)

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt