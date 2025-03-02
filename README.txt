Customer Segmentation using K-Means

Overview
This project applies the K-Means clustering algorithm to segment customers based on their purchase behavior. The application is built using Python and Streamlit, allowing users to upload a dataset, visualize customer segments, and download the clustered data.

Features
- Upload a CSV dataset
- Encode categorical variables
- Normalize data for better clustering
- Determine the optimal number of clusters using the Elbow method
- Apply K-Means clustering to segment customers
- Visualize customer segments using PCA
- Download the segmented customer data

Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

Installation
1. Clone the Repository

git clone https://github.com/your-repo/customer-segmentation.git
cd customer-segmentation


2. Install Dependencies

pip install streamlit pandas numpy matplotlib seaborn scikit-learn


3. Run the Application
streamlit run app.py

or

<full_path_of_stremlit> run app.py


Usage
1. Upload a CSV file containing customer data.
2. View a preview of the dataset.
3. Determine the optimal number of clusters using the Elbow method.
4. Select the number of clusters (K) using a slider.
5. View customer segments visualized in a scatter plot.
6. Download the segmented data as a CSV file.




