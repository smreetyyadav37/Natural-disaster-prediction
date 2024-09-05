import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
df = pd.read_csv("processed_earthquake_data.csv")

# Plot histograms for each feature
plt.figure(figsize=(12, 10))
df[['latitude', 'longitude', 'depth', 'magnitude']].hist(bins=50, edgecolor='black', layout=(2, 2), figsize=(12, 10))
plt.suptitle('Histograms of Earthquake Features')
plt.savefig('EDA_Plots/histograms.png')
plt.show()

# Plot scatter plots to visualize pairwise relationships
pairplot = sns.pairplot(df[['latitude', 'longitude', 'depth', 'magnitude']])
pairplot.fig.suptitle('Pairwise Relationships of Earthquake Features', y=1.02)
pairplot.savefig('EDA_Plots/pairwise_relationships.png')
plt.show()

# Plot correlation matrix with heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[['latitude', 'longitude', 'depth', 'magnitude']].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Earthquake Features')
plt.savefig('EDA_Plots/correlation_matrix.png')
plt.show()

plt.figure(figsize=(12, 10))
sns.boxplot(data=df[['latitude', 'longitude', 'depth', 'magnitude']])
plt.title('Box Plots of Earthquake Features')
plt.savefig('EDA_Plots/boxplot.png')
plt.show()



