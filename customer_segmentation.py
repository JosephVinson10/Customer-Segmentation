# customer_segmentation_ml.py

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html

# Step 1: Load and Prepare the Data
def load_data(filename):
    try:
        data = pd.read_csv(filename)
        data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
        return data
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        exit()

# Step 2: RFM Analysis
def perform_rfm_analysis(data):
    current_date = datetime(2024, 10, 29)
    rfm = data.groupby('CustomerName').agg({
        'TransactionDate': lambda x: (current_date - x.max()).days,  # Recency
        'TransactionID': 'count',  # Frequency
        'AmountSpent': 'sum',  # Monetary Value
        'ProductName': lambda x: ', '.join(x.unique())  # Products Purchased
    }).reset_index()
    rfm.columns = ['CustomerName', 'Recency', 'Frequency', 'Monetary', 'ProductsPurchased']
    return rfm

# Step 3: Standardize the Data
def scale_data(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm_scaled

# Step 4: Apply Clustering and Generate Dynamic Labels
def apply_clustering(rfm_scaled, rfm, model_type, n_clusters=6):
    if model_type == 'kmeans':
        print("Running K-Means...")
        model = KMeans(n_clusters=n_clusters, random_state=42)
        rfm['Cluster'] = model.fit_predict(rfm_scaled)
        score = silhouette_score(rfm_scaled, rfm['Cluster'])
        print(f'Silhouette Score for K-Means: {score:.2f}')

    elif model_type == 'dbscan':
        print("Running DBSCAN...")
        model = DBSCAN(eps=0.5, min_samples=5)
        rfm['Cluster'] = model.fit_predict(rfm_scaled)

        if (rfm['Cluster'] == -1).all():
            print("Warning: DBSCAN found only noise. Try adjusting parameters.")
        elif len(rfm['Cluster'].unique()) <= 1:
            print("DBSCAN found insufficient clusters. Adjust eps or min_samples.")

    return rfm

# Step 5: Generate Dynamic Cluster Explanations
def generate_cluster_explanations(rfm):
    cluster_descriptions = {}
    unique_clusters = sorted(rfm['Cluster'].unique())

    for cluster in unique_clusters:
        if cluster == -1:
            description = "Noise or outliers (DBSCAN only)."
        else:
            # Dynamically create explanations based on average values in the cluster
            cluster_data = rfm[rfm['Cluster'] == cluster]
            avg_recency = cluster_data['Recency'].mean()
            avg_frequency = cluster_data['Frequency'].mean()
            avg_monetary = cluster_data['Monetary'].mean()

            description = (
                f"Cluster {cluster}: Avg Recency: {avg_recency:.2f} days, "
                f"Avg Frequency: {avg_frequency:.2f} transactions, "
                f"Avg Monetary: ${avg_monetary:.2f}"
            )
        cluster_descriptions[cluster] = description

    return cluster_descriptions

# Step 6: Create Dashboard with Dynamic Explanations
def create_dashboard(rfm, cluster_descriptions):
    app = dash.Dash(__name__)

    # 3D Scatter Plot
    fig_3d = px.scatter_3d(
        rfm,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster',
        hover_name='CustomerName',
        hover_data={'ProductsPurchased': True},
        title='3D Customer Segmentation (Recency vs Frequency vs Monetary)',
        labels={
            'Recency': 'Days since last purchase',
            'Frequency': 'Number of Transactions',
            'Monetary': 'Total Spend in USD'
        }
    )

    # 2D Scatter Plot
    fig_2d = px.scatter(
        rfm,
        x='Frequency',
        y='Monetary',
        color='Cluster',
        hover_name='CustomerName',
        hover_data={'ProductsPurchased': True},
        title='Frequency vs Monetary Value',
        labels={
            'Frequency': 'Number of Transactions',
            'Monetary': 'Total Spend in USD'
        }
    )

    # Summary Table
    summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=list(summary.columns), fill_color='paleturquoise', align='left'),
        cells=dict(values=[summary[col] for col in summary.columns], fill_color='lavender', align='left')
    )])

    # Dashboard Layout with Dynamic Cluster Explanations
    app.layout = html.Div([
        html.H1('Customer Segmentation Dashboard'),
        dcc.Graph(figure=fig_3d),
        dcc.Graph(figure=fig_2d),
        dcc.Graph(figure=fig_table),
        html.H2('Cluster Explanations'),
        html.Ul([html.Li(desc) for desc in cluster_descriptions.values()])
    ])

    return app

# Main Function to Run the Pipeline
def main():
    # Load Data
    data = load_data('customer_transactions_with_products.csv')

    # Perform RFM Analysis
    rfm = perform_rfm_analysis(data)

    # Scale Data
    rfm_scaled = scale_data(rfm)

    # Select Model and Apply Clustering
    model_type = input("Choose clustering model (kmeans/dbscan): ").strip().lower()
    if model_type not in ['kmeans', 'dbscan']:
        print("Invalid input! Defaulting to K-Means.")
        model_type = 'kmeans'

    n_clusters = 6  # Default for K-Means
    if model_type == 'kmeans':
        n_clusters = int(input("Enter the number of clusters: "))

    # Apply Clustering
    rfm = apply_clustering(rfm_scaled, rfm, model_type, n_clusters)

    # Generate Dynamic Explanations
    cluster_descriptions = generate_cluster_explanations(rfm)

    # Launch Dashboard
    print("Launching the dashboard...")
    app = create_dashboard(rfm, cluster_descriptions)
    app.run_server(debug=True)

# Entry Point
if __name__ == '__main__':
    main()
