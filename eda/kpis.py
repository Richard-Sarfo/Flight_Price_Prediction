def avg_fare_per_airline(df):
    return (
        df.groupby('Airline')['Total Fare (BDT)']
        .mean()
        .sort_values(ascending=False)
    )

def most_popular_route(df):
    return (
        df.groupby(['Source', 'Destination'])
        .size()
        .sort_values(ascending=False)
        .reset_index(name='Flight Count')
        .head(1)
    )

def seasonal_fare_variation(df):
    return (
        df.groupby('Seasonality')['Total Fare (BDT)']
        .agg(['mean', 'min', 'max'])
        .sort_values('mean', ascending=False)
    )

def top_expensive_routes(df, top_n=5):
    return (
        df.groupby(['Source', 'Destination'])['Total Fare (BDT)']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
