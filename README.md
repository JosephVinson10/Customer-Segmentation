# **Customer Segmentation Using K-Means and DBSCAN**

This project performs **customer segmentation** using both **K-Means** and **DBSCAN** clustering algorithms, helping group customers based on their behavior. Through this project, I learned how to use **Recency-Frequency-Monetary (RFM) analysis** to classify customers into meaningful segments. The project also showcases an interactive **dashboard** built with **Plotly Dash** to explore clustering results.

---

## **Features**

- **RFM Analysis**: Segments customers based on:
  - **Recency**: Days since the last purchase.
  - **Frequency**: Number of transactions.
  - **Monetary Value**: Total amount spent.

- **Clustering Models**:
  - **K-Means**: Allows specification of the number of clusters.
  - **DBSCAN**: Detects clusters based on density, with noise handling.

- **Dynamic Cluster Explanations**:
  - Generates detailed descriptions for each cluster based on the average **Recency**, **Frequency**, and **Monetary** values.

- **Interactive Dashboard**:
  - **3D Scatter Plot**: Visualizes segmentation in 3 dimensions (RFM).
  - **2D Scatter Plot**: Displays clusters based on frequency and monetary value.
  - **Summary Table**: Aggregates statistics for each cluster.
  - **Cluster Legend**: Automatically generated explanations for clusters.

---

## **Project Structure**

```
.
├── customer_transactions_with_products.csv   # Sample dataset used in the project
├── customer_segmentation_ml.py               # Main Python code for segmentation
├── README.md                                 # This README file
```

---

## **How to Run the Project**

### **1. Prerequisites**

- **Python 3.x** installed on your system.
- Install the required libraries by running:

  ```bash
  pip install pandas numpy scikit-learn plotly dash
  ```

### **2. Run the Code**

1. Clone the repository or download the code files.
2. Ensure the `customer_transactions_with_products.csv` is in the same directory as the Python code.
3. Open a terminal/command prompt and run:

   ```bash
   python customer_segmentation_ml.py
   ```

4. **Select the clustering model** when prompted:
   - Enter `kmeans` or `dbscan`.
   - If you choose K-Means, enter the **desired number of clusters**.

5. **Access the Dashboard**:
   - Once the server starts, open your browser and go to:
     ```
     http://127.0.0.1:8050/
     ```

---

## **How the Dashboard Works**

- **3D Scatter Plot**: 
  - Visualizes the segmentation with Recency, Frequency, and Monetary values.
  - Hover over points to see customer details and products purchased.

- **2D Scatter Plot**: 
  - Displays segmentation based on **Frequency vs Monetary Value**.

- **Summary Table**:
  - Shows the **mean Recency, Frequency, and Monetary Value** for each cluster.

- **Cluster Legend**:
  - Provides dynamic explanations for each cluster based on clustering output.

---

## **Sample Dataset**

The **`customer_transactions_with_products.csv`** file contains:

- **TransactionID**: Unique ID for each transaction.
- **CustomerName**: Name of the customer.
- **TransactionDate**: Date of the transaction.
- **AmountSpent**: Amount spent in the transaction.
- **ProductName**: Product purchased.

---

## **Project Workflow**

1. **Load Data**:
   - Load transaction data from the CSV file.

2. **RFM Analysis**:
   - Calculate Recency, Frequency, and Monetary metrics for each customer.

3. **Clustering**:
   - Apply K-Means or DBSCAN to segment customers.
   - Generate dynamic explanations for each cluster based on the clustering results.

4. **Dashboard**:
   - Visualize the results in an interactive Plotly Dash dashboard.

---

## **Customization**

- **Modify Clustering Parameters**:
  - Adjust the **`eps`** and **`min_samples`** values in DBSCAN for better clustering.
  - Change the **number of clusters** for K-Means based on needs.

- **Enhance Product Data**:
  - Add more product names to the dataset to make it more relevant.

---

## **Future Improvements**

- **Customer Lifetime Value (CLV)**: Add CLV predictions for better segmentation.
- **Model Comparison**: Compare K-Means and DBSCAN with other algorithms.
- **Automated Reporting**: Generate PDF reports of segmentation results.
