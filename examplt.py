import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

test_data = pd.read_csv(r'data\raw\test_data.csv')
label = pd.read_csv(r'C:\MB_Project\project\Competition\Anomaly-Detection-of-Air-Compressor\data\submission\submission.csv')
test_data["label"] = label['label']

print(test_data.label.value_counts())

test1 = test_data[test_data["label"] == 1]
# sns.pairplot(test1[['air_inflow', 'air_end_temp', 'out_pressure']])
sns.pairplot(test1[['motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']])
plt.show()
