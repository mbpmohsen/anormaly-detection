import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

data = {
    'timestamp': ['2024-10-03 14:23:45', '2024-10-03 14:25:12', '2024-10-03 14:27:32', '2024-10-03 14:30:45'],
    'log_level': ['INFO', 'ERROR', 'INFO', 'INFO'],
    'message': ['User logged in', 'System failure', 'File uploaded', 'User logged out']
}

df = pd.DataFrame(data)

print("Original Data:")
print(df.head())

df['log_level_encoded'] = df['log_level'].map({'INFO': 0, 'ERROR': 1})

model = IsolationForest(contamination=0.25)
df['anomaly'] = model.fit_predict(df[['log_level_encoded']])

print("\nAnomaly Detection Results:")
print(df[['timestamp', 'log_level', 'anomaly']])

plt.scatter(df.index, df['log_level_encoded'], c=df['anomaly'], cmap='coolwarm')
plt.xlabel('Log Entry')
plt.ylabel('Log Level (0 = INFO, 1 = ERROR)')
plt.title('Anomaly Detection in Log Files')
plt.show()
