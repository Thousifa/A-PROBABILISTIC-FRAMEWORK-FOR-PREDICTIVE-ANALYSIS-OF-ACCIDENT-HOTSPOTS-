import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('accident_data.csv')  # Replace with your actual file path

# Encode categorical features
label_encoders = {}
for column in ['Accident_Severity', 'Light_Conditions', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Define Bayesian Network structure
bayesian_model = BayesianNetwork([
    ('Light_Conditions', 'Accident_Severity'),
    ('Weather_Conditions', 'Accident_Severity'),
    ('Road_Type', 'Accident_Severity'),
    ('Urban_or_Rural_Area', 'Accident_Severity'),
    ('Vehicle_Type', 'Accident_Severity'),
])

# Fit the model
bayesian_model.fit(df, estimator=MaximumLikelihoodEstimator)

# Perform inference to predict accident severity or hotspot probability under given conditions
inference = VariableElimination(bayesian_model)
query_result = inference.query(variables=['Accident_Severity'], 
                               evidence={'Light_Conditions': 1, 'Weather_Conditions': 0, 'Road_Type': 2, 'Urban_or_Rural_Area': 0, 'Vehicle_Type': 1})
print(query_result)
