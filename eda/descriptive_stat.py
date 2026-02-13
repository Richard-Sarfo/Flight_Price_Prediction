def summarize_fares(df, group_cols, fare_col='Total Fare (BDT)'):
    """
    Summarize fare statistics by given grouping columns
    """
    return (
        df.groupby(group_cols)[fare_col]
        .agg(['count', 'mean', 'median', 'min', 'max', 'std'])
        .reset_index()
    )

def numerical_correlation(df):
    """
    Compute correlation matrix for numerical columns
    """
    numeric_df = df.select_dtypes(include='number')
    return numeric_df.corr()
