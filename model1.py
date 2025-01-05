import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('accident_data.csv') 

# Preprocess categorical data
label_encoders = {}
for column in ['Accident_Severity', 'Light_Conditions', 'District Area', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Define Bayesian network structure
bayesian_model = BayesianNetwork([
    ('Weather_Conditions', 'Accident_Severity'),
    ('Light_Conditions', 'Accident_Severity'),
    ('Road_Surface_Conditions', 'Accident_Severity'),
    ('Road_Type', 'Accident_Severity'),
    ('Vehicle_Type', 'Accident_Severity'),
    ('Urban_or_Rural_Area', 'Accident_Severity'),
])

# Fit the model with Maximum Likelihood Estimator
bayesian_model.fit(df, estimator=MaximumLikelihoodEstimator)

# Make inference
inference = VariableElimination(bayesian_model)
query_result = inference.query(variables=['Accident_Severity'], 
                               evidence={'Weather_Conditions': 0, 'Light_Conditions': 1, 'Road_Type': 2})
print(query_result)
