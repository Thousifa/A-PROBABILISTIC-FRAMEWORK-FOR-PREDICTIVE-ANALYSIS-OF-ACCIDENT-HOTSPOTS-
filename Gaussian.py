import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('accident_data.csv')  # Replace with your actual file path

# Encode categorical features
label_encoders = {}
for column in ['Light_Conditions', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Select features for clustering
features = df[['Latitude', 'Longitude', 'Light_Conditions', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type']]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

# Define and fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
gmm.fit(features)

# Predict clusters (hotspots)
hotspot_clusters = gmm.predict(features)
df['Hotspot_Cluster'] = hotspot_clusters

# Calculate the probability of each location being in a specific hotspot cluster
df['Hotspot_Probability'] = gmm.predict_proba(features).max(axis=1)

# Display results
print(df[['Latitude', 'Longitude', 'Hotspot_Cluster', 'Hotspot_Probability']].head())
