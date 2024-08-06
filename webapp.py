import warnings
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
import streamlit as st
import numpy as np

# Suppress warnings related to date parsing
warnings.filterwarnings("ignore")


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if isinstance(df[column].dtype, pd.CategoricalDtype) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                ).lower()
                if user_text_input:
                    df = df[df[column].astype(str).str.lower().str.contains(user_text_input)]

    return df

def get_highest_rating_price(names, df):
    final = []
    for name in names:
        median_price = round(df[(df['Menu'] == name) & (df['Rating'] >= 4.8) & (df['Review'] >= 1000)]['Price'].median(),2)
        final.append((name, median_price))
    
    return final

df = pd.read_csv("clean_data.csv")
df_elastic = pd.read_excel('item_elasticity.xlsx')

grouped = df.groupby(['Menu', 'Category']).agg({'Price': 'median'}).round(2)
grouped.reset_index(inplace=True)
grouped.columns = ['Menú', 'Categoria', 'Preço médio (€)']

names = list(grouped.Menú)

high_rated = get_highest_rating_price(names, df)
high_rated_df = pd.DataFrame(high_rated, columns=['Menú', 'Preço Médio de Alta Classificação'])

merged = grouped.merge(high_rated_df)

# Create a dictionary for quick lookup
elasticity_dict = dict(zip(df_elastic['Item'].str.lower(), df_elastic['Classification']))
# Use lambda function to create the new 'Elasticity' column
merged['Elasticity'] = merged['Menú'].apply(lambda x: next((elasticity_dict[item] for item in elasticity_dict if item in x.lower()), 'Unknown'))

def calculate_ideal_prices(row):
    # Calculate Preço Ideal
    if row['Preço Médio'] <= np.floor(row['Preço Médio']) + 0.50:
        preco_ideal = np.floor(row['Preço Médio']) + 0.59
    else:
        preco_ideal = np.floor(row['Preço Médio']) + 0.99
    
    # Calculate Preço Ideal de Alta Classificação
    if pd.notna(row['Preço Médio de Alta Classificação']):
        preco_ideal_alta = np.floor(row['Preço Médio de Alta Classificação']) + 0.99
    else:
        preco_ideal_alta = np.nan
    
    return pd.Series({'Preço Ideal': preco_ideal, 'Preço Ideal de Alta Classificação': preco_ideal_alta})

merged[['Preço Ideal', 'Preço Ideal de Alta Classificação']] = merged.apply(calculate_ideal_prices, axis=1)
final_table = merged[['Menú', 'Categoria', 'Preço Médio', 'Preço Ideal', 'Preço Médio de Alta Classificação', 'Preço Ideal de Alta Classificação']]

st.title('Análise de Preço Médio dos Restaurantes em Portugal')

st.dataframe(filter_dataframe(final_table), hide_index=True)

