import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# File paths
emissions_file = 'annual-co2-emissions-per-country.csv'
gdp_population_file = 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_73467.csv'

# Step 1: Data Acquisition and Exploration

# Load CO2 emissions data
emissions_data = pd.read_csv(emissions_file)
emissions_data = emissions_data.rename(columns={"Entity": "Country", "Year": "Year", "Annual CO2 emissions": "CO2_Emissions"})

# Load GDP and population data
gdp_data = pd.read_csv(gdp_population_file, skiprows=4)
gdp_data = gdp_data.rename(columns={"Country Name": "Country"})

print(emissions_data.head)
print(gdp_data.head)

# Melt GDP data for year-wise analysis
gdp_data_melted = gdp_data.melt(
    id_vars=["Country", "Country Code", "Indicator Name", "Indicator Code"],
    var_name="Year",
    value_name="GDP_Per_Capita"
)

# Filter out invalid years and convert to integers
gdp_data_melted = gdp_data_melted[gdp_data_melted["Year"].str.isnumeric()]
gdp_data_melted["Year"] = gdp_data_melted["Year"].astype(int)


# Merge datasets
print("Merging datasets...")
merged_data = pd.merge(
    emissions_data[["Country", "Year", "CO2_Emissions"]],
    gdp_data_melted[["Country", "Year", "GDP_Per_Capita"]],
    on=["Country", "Year"],
    how="inner"
)

# Step 2: Data Preparation
print("Cleaning and preparing data...")

# Identify missing values
print("\nChecking for missing values:")
print(merged_data.isnull().sum())
merged_data = merged_data.dropna()
merged_data["CO2_Emissions_Per_Capita"] = merged_data["CO2_Emissions"] / 1000000
merged_data["GDP_Per_Capita"] = merged_data["GDP_Per_Capita"] / 1000  # Scale GDP for visualization


# Basic statistics of the dataset
print("\nDataset Summary:")
print(merged_data.describe())


# Correlation Matrix: Select only numeric columns
print("\nCorrelation Matrix (numeric columns only):")
numeric_columns = merged_data.select_dtypes(include=[np.number])
print(numeric_columns.corr())


# Visualization 1: Scatter Matrix Plot
print("\nCreating scatter matrix plot...")
scatter_matrix_columns = ["CO2_Emissions_Per_Capita", "GDP_Per_Capita", "CO2_Emissions"]
scatter_matrix_fig = pd.plotting.scatter_matrix(
    merged_data[scatter_matrix_columns],
    figsize=(8, 8),
    alpha=0.7,
    diagonal='kde'
)
plt.suptitle("Scatter Matrix: CO2 Emissions and GDP Per Capita")
plt.show()

# Visualization 2: Histograms
print("\nCreating histograms...")
merged_data[scatter_matrix_columns].hist(bins=20, figsize=(10, 6), edgecolor='black')
plt.suptitle("Histograms of CO2 Emissions and GDP Per Capita")
plt.show()

# Visualization 3: Heatmap
print("\nCreating heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(
    merged_data.select_dtypes(include=[np.number]).corr(),  # Use only numeric columns
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap: CO2 Emissions and GDP Per Capita")
plt.show()

# Visualization 4 (Creative Visualization): CO2 Emissions Over Time
print("\nCreating creative visualization: CO2 emissions over time...")
plt.figure(figsize=(10, 6))
for country in merged_data["Country"].unique()[:10]:  # Limit to 10 countries for clarity
    country_data = merged_data[merged_data["Country"] == country]
    plt.plot(country_data["Year"], country_data["CO2_Emissions"], label=country)
plt.xlabel("Year")
plt.ylabel("CO2 Emissions (tonnes)")
plt.title("CO2 Emissions Over Time for Selected Countries")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
plt.tight_layout()
plt.show()


# Step 2: Data Preparation
print("\nPreparing the dataset...")

# 1. Handle Missing Values
# Fill missing numerical values with the median
for column in merged_data.select_dtypes(include=[np.number]).columns:
    if merged_data[column].isnull().sum() > 0:
        merged_data[column].fillna(merged_data[column].median(), inplace=True)

# Fill missing categorical values with the mode
for column in merged_data.select_dtypes(include=[object]).columns:
    if merged_data[column].isnull().sum() > 0:
        merged_data[column].fillna(merged_data[column].mode()[0], inplace=True)

# 2. Normalize Numerical Features
scaler = MinMaxScaler()
numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_columns] = scaler.fit_transform(merged_data[numeric_columns])

# 3. Encode Categorical Variables
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop one to avoid multicollinearity
categorical_columns = [col for col in merged_data.select_dtypes(include=[object]).columns if col != "Country"]

# Apply one-hot encoding
encoded_features = encoder.fit_transform(merged_data[categorical_columns])
encoded_feature_names = encoder.get_feature_names_out(categorical_columns)

# Add encoded features to the DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=merged_data.index)
merged_data = pd.concat([merged_data, encoded_df], axis=1)

# Drop original categorical columns
merged_data.drop(columns=categorical_columns, inplace=True)

# 4. Engineer New Features
# Feature 1: Emissions Intensity (Emissions per GDP unit)
merged_data["Emissions_Intensity"] = merged_data["CO2_Emissions"] / (merged_data["GDP_Per_Capita"] + 1e-6)

# Feature 2: GDP Growth Rate (Percentage change from previous year)
merged_data["GDP_Growth_Rate"] = merged_data.groupby("Country")["GDP_Per_Capita"].pct_change().fillna(0)

# Display prepared data
print("\nPrepared Dataset:")
print(merged_data.head())


#Step 3: Advanced Visualization Development

# Visualization 1: Clustered Scatter Plot
print("\nCreating clustered scatter plot...")
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    merged_data["GDP_Per_Capita"],
    merged_data["CO2_Emissions_Per_Capita"],
    c=merged_data["Emissions_Intensity"],  # Color by Emissions Intensity
    cmap="viridis",
    alpha=0.7
)
plt.colorbar(scatter, label="Emissions Intensity")
plt.xlabel("GDP Per Capita (scaled)")
plt.ylabel("CO2 Emissions Per Capita (scaled)")
plt.title("Clustered Scatter Plot: GDP vs. CO2 Emissions Per Capita")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization 2: 3D Scatter Plot
print("\nCreating 3D visualization...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    merged_data["GDP_Per_Capita"],
    merged_data["CO2_Emissions_Per_Capita"],
    merged_data["Emissions_Intensity"],
    c=merged_data["GDP_Growth_Rate"],  # Color by GDP Growth Rate
    cmap="coolwarm",
    alpha=0.7
)
ax.set_xlabel("GDP Per Capita (scaled)")
ax.set_ylabel("CO2 Emissions Per Capita (scaled)")
ax.set_zlabel("Emissions Intensity")
ax.set_title("3D Scatter Plot: GDP, CO2 Emissions, and Emissions Intensity")
plt.colorbar(scatter, label="GDP Growth Rate", shrink=0.5, aspect=10)
plt.show()



