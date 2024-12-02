from nixtla import NixtlaClient
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Initialize the TimeGPT client with your API key
nixtla_client = NixtlaClient(api_key='YOUR-API-KEY')

# Load the dataset containing electric production data and pre-process data
df = pd.read_csv('Electric_Production.csv')
df.rename(columns={'DATE': 'date', 'IPG2211A2N': 'power'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Split the data into training and testing sets
train_cutoff = '2015-12-31'
df_train = df[df['date'] <= '2015-12-31']
df_test = df[df['date'].between('2016-01-01', '2017-12-31')]

# Generate forecasts without fine-tuning (baseline)
preds_no_ft = nixtla_client.forecast(
    df=df_train, h=24,
    time_col='date', target_col='power',
)

# Generate forecasts with 50 fine-tuning steps
preds_ft_50_steps = nixtla_client.forecast(
    df=df_train, h=24, finetune_steps=50,
    time_col='date', target_col='power',
)

# Generate forecasts with 500 fine-tuning steps
preds_ft_500_steps = nixtla_client.forecast(
    df=df_train, h=24, finetune_steps=500,
    time_col='date', target_col='power',
)

# Evaluate forecast performance using Mean Absolute Error (MAE)
print(mean_absolute_error(df_test['power'], preds_no_ft["TimeGPT"]))
print(mean_absolute_error(df_test['power'], preds_ft_50_steps["TimeGPT"]))
print(mean_absolute_error(df_test['power'], preds_ft_500_steps["TimeGPT"]))
