import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("motor_temp.csv")

sns.heatmap(data.corr(), annot=True)
plt.show()

sns.scatterplot(x="motor_load", y="stator_temp", data=data)
plt.show()