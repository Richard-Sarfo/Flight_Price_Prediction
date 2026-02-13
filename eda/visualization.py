import matplotlib.pyplot as plt
import seaborn as sns

def plot_fare_distributions(df):
    cols = ['Total Fare (BDT)', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)']
    
    for col in cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

def boxplot_fare_by_airline(df):
    plt.figure(figsize=(10,5))
    sns.boxplot(x='Airline', y='Total Fare (BDT)', data=df)
    plt.xticks(rotation=45)
    plt.title('Fare Variation Across Airlines')
    plt.show()

def plot_avg_fare_over_time(df, time_col):
    avg_fare = df.groupby(time_col)['Total Fare (BDT)'].mean()
    
    plt.figure(figsize=(8,4))
    avg_fare.plot(marker='o')
    plt.ylabel('Average Fare')
    plt.title(f'Average Fare by {time_col}')
    plt.grid(True)
    plt.show()

def correlation_heatmap(df):
    corr = df.select_dtypes(include='number').corr()
    
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')

    plt.show()