"""
Visualization module for Whole Foods supply chain analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict
import folium
from folium.plugins import HeatMap
import os
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_visualizations(processed_data: Dict, analysis_results: Dict):
    """Create all visualizations for the analysis."""
    print("Creating visualizations...")

    df = processed_data['merged']

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # VISUALIZATION 1: Bar chart - Average sales by department
    if 'department' in df.columns:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        dept_sales = df.groupby('department')['sales_volume'].mean().sort_values()
        colors = plt.cm.Set3(np.linspace(0, 1, len(dept_sales)))
        dept_sales.plot(kind='barh', ax=ax1, color=colors)
        ax1.set_title('Average Sales Volume by Department', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Average Sales Volume', fontsize=12)
        ax1.set_ylabel('Department', fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/avg_sales_by_department.png', dpi=300, bbox_inches='tight')
        print("✓ Created: avg_sales_by_department.png")

    # VISUALIZATION 2: Line chart - Sales trends by month
    if 'month' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        monthly_sales = df.groupby('month')['sales_volume'].mean()
        ax2.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, markersize=8)
        ax2.set_title('Monthly Sales Trends', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Average Sales Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(1, 13))
        plt.tight_layout()
        plt.savefig('outputs/monthly_sales_trends.png', dpi=300, bbox_inches='tight')
        print("✓ Created: monthly_sales_trends.png")

    # VISUALIZATION 3: Heatmap - Sales by department and day of week
    if 'department' in df.columns and 'day_name' in df.columns:
        fig3, ax3 = plt.subplots(figsize=(12, 8))

        # Aggregate data
        heatmap_data = df.groupby(['department', 'day_name']).agg({
            'sales_volume': 'mean'
        }).reset_index()

        # Pivot for heatmap
        pivot_data = heatmap_data.pivot(index='department', columns='day_name', values='sales_volume')

        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex(columns=day_order)

        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd',
                    ax=ax3, cbar_kws={'label': 'Average Sales'})
        ax3.set_title('Heatmap: Sales by Department and Day of Week',
                      fontsize=16, fontweight='bold')
        ax3.set_xlabel('Day of Week', fontsize=12)
        ax3.set_ylabel('Department', fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/sales_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Created: sales_heatmap.png")

    # VISUALIZATION 4: Correlation matrix heatmap
    if 'correlation_matrix' in analysis_results:
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        correlation_matrix = analysis_results['correlation_matrix']
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True, ax=ax4,
                    cbar_kws={'shrink': 0.8})
        ax4.set_title('Correlation Matrix of Key Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Created: correlation_matrix.png")

    # VISUALIZATION 5: Box plot - Waste percentage by department
    if 'department' in df.columns:
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        df.boxplot(column='waste_percentage', by='department', ax=ax5, grid=False)
        ax5.set_title('Waste Percentage Distribution by Department', fontsize=16, fontweight='bold')
        ax5.set_xlabel('Department', fontsize=12)
        ax5.set_ylabel('Waste Percentage', fontsize=12)
        plt.suptitle('')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('outputs/waste_by_department.png', dpi=300, bbox_inches='tight')
        print("✓ Created: waste_by_department.png")

    # VISUALIZATION 6: Feature importance (if available)
    if 'feature_importance' in analysis_results and analysis_results['feature_importance'] is not None:
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        feature_importance = analysis_results['feature_importance']
        feature_importance.sort_values('importance', ascending=True).plot(
            kind='barh', x='feature', y='importance', ax=ax6, color='darkgreen'
        )
        ax6.set_title('Feature Importance for Waste Prediction (Random Forest)',
                      fontsize=16, fontweight='bold')
        ax6.set_xlabel('Importance Score', fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Created: feature_importance.png")

    # VISUALIZATION 7: Interactive scatter plot (Plotly)
    if 'inventory_level' in df.columns and 'sales_volume' in df.columns:
        try:
            fig7 = px.scatter(df, x='inventory_level', y='sales_volume',
                              color='department', size='waste_percentage',
                              hover_data=['store', 'restock_time_days'],
                              title='Inventory vs Sales with Waste Percentage',
                              labels={'inventory_level': 'Inventory Level',
                                      'sales_volume': 'Sales Volume',
                                      'department': 'Department'},
                              template='plotly_white')
            fig7.write_html('outputs/interactive_scatter.html')
            print("✓ Created: interactive_scatter.html")
        except:
            print("Note: Plotly visualization skipped")

    # VISUALIZATION 8: Pie chart - Department distribution
    if 'department' in df.columns:
        fig8, ax8 = plt.subplots(figsize=(10, 8))
        dept_dist = df['department'].value_counts()
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(dept_dist)))
        ax8.pie(dept_dist.values, labels=dept_dist.index, autopct='%1.1f%%',
                startangle=90, colors=colors)
        ax8.set_title('Department Distribution in Dataset', fontsize=16, fontweight='bold')
        ax8.axis('equal')
        plt.tight_layout()
        plt.savefig('outputs/department_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Created: department_distribution.png")

    print(f"\n✓ All visualizations saved in 'outputs/' folder")

    # Show one plot
    plt.show()


def create_interactive_dashboard(processed_data: Dict):
    """Create an interactive Folium map dashboard."""
    print("\nCreating interactive dashboard...")

    df = processed_data['merged']

    # Create a base map centered on US
    us_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4,
                        tiles='CartoDB positron', control_scale=True)

    # Define coordinates for store locations
    store_coordinates = {
        'NYC_Chelsea': [40.7466, -74.0018],
        'SF_Noe_Valley': [37.7509, -122.4310],
        'LA_Venice': [33.9961, -118.4811],
        'CHI_Lincoln_Park': [41.9217, -87.6487],
        'TX_Austin': [30.2672, -97.7431]
    }

    # Check if store data is available
    if 'store' in df.columns:
        # Calculate store statistics
        store_stats = df.groupby('store').agg({
            'sales_volume': 'mean',
            'waste_percentage': 'mean',
            'restock_time_days': 'mean',
            'inventory_level': 'mean'
        }).reset_index()

        # Add markers for each store
        for _, row in store_stats.iterrows():
            store = row['store']
            coords = store_coordinates.get(store, [39.8283, -98.5795])

            # Create popup content
            popup_content = f"""
            <div style="font-family: Arial; width: 220px;">
                <h4>{store.replace('_', ' ')}</h4>
                <hr style="margin: 5px 0;">
                <p><b>Avg Sales:</b> {row['sales_volume']:,.0f}</p>
                <p><b>Waste %:</b> {row['waste_percentage']:.2f}%</p>
                <p><b>Restock Time:</b> {row['restock_time_days']:.1f} days</p>
                <p><b>Avg Inventory:</b> {row['inventory_level']:,.0f}</p>
            </div>
            """

            # Color code based on waste percentage
            waste = row['waste_percentage']
            if waste > 3:
                color = 'red'
            elif waste > 1.5:
                color = 'orange'
            else:
                color = 'green'

            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{store.replace('_', ' ')} (Click for details)",
                icon=folium.Icon(color=color, icon='shopping-cart', prefix='fa')
            ).add_to(us_map)

        # Add a heatmap for sales volume
        heat_data = []
        for store, coords in store_coordinates.items():
            if store in store_stats['store'].values:
                store_data = store_stats[store_stats['store'] == store]
                if not store_data.empty:
                    store_sales = store_data['sales_volume'].values[0]
                    heat_data.append([coords[0], coords[1], store_sales / 1000])

        if heat_data:
            HeatMap(heat_data, radius=25, blur=15, max_zoom=1,
                    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'red'}).add_to(us_map)
    else:
        # Add default markers if no store data
        for store, coords in store_coordinates.items():
            folium.Marker(
                location=coords,
                popup=f"<b>{store.replace('_', ' ')}</b><br>Sample Whole Foods Store",
                icon=folium.Icon(color='blue', icon='store', prefix='fa')
            ).add_to(us_map)

    # Add title to map
    title_html = '''
    <h3 align="center" style="font-size:16px"><b>Whole Foods Store Performance Dashboard</b></h3>
    '''
    us_map.get_root().html.add_child(folium.Element(title_html))

    # Save the map
    us_map.save('dashboard.html')
    print("✓ Interactive dashboard saved as 'dashboard.html'")
    print("  Open this file in a web browser to view the interactive map")