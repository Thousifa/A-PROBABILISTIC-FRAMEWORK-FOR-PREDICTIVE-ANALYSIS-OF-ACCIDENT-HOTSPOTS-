import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('accident_data.csv')  # Replace with your actual file path

# Prepare data by sorting by date
df['Accident Date'] = pd.to_datetime(df['Accident Date'], dayfirst=True)
df = df.sort_values('Accident Date')

# Encode categorical features
label_encoders = {}
for column in ['Light_Conditions', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Prepare sequence data
sequence_data = df[['Light_Conditions', 'Weather_Conditions', 'Road_Type', 'Urban_or_Rural_Area', 'Vehicle_Type']].values

# Define and fit Hidden Markov Model
hmm_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
hmm_model.fit(sequence_data)

# Predict hidden states
hidden_states = hmm_model.predict(sequence_data)
df['Hidden_State'] = hidden_states

# Display accident sequence and identified patterns
print(df[['Accident Date', 'Light_Conditions', 'Weather_Conditions', 'Road_Type', 'Urban_or_Rural_Area', 'Vehicle_Type', 'Hidden_State']].head())
