
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Configurações iniciais
st.set_page_config(page_title="Previsão de Vendas", layout="centered")
st.title("Previsão de Vendas")
st.markdown("Visualização e previsão de séries temporais referentes ao mercado de jogos")

# Carregar os dados
@st.cache_data
def load_data():
    return pd.read_csv("gaming_data/gaming_industry_trends.csv")

df = load_data()

# Menu principal
menu = st.radio("Escolha uma opção:", ["Previsão por Plataforma", "Previsão por Gênero"])

# Função de processamento e predição
def process_and_predict(df, group_by, selected_value):
    other_col = 'Genre' if group_by == 'Platform' else 'Platform'
    grouped_df = df.groupby([group_by, 'Release Year'])
    new_data = []

    for (key, year), group in grouped_df:
        group_data = {group_by: key, 'Release Year': year}
        text_columns = ['Developer', other_col, 'Esports Popularity', 'Trending Status']
        group[text_columns] = group[text_columns].astype(str)
        group['combined_text'] = group[text_columns].apply(lambda row: ' '.join(row), axis=1)
        vectorizer = CountVectorizer()
        vectorized_data = vectorizer.fit_transform(group['combined_text'])
        feature_names = vectorizer.get_feature_names_out()

        for i, feature in enumerate(feature_names):
            try:
                group_data[feature] = vectorized_data.toarray()[0][i]
            except:
                group_data[feature] = 0

        for col in ['Players (Millions)', 'Peak Concurrent Players', 'Metacritic Score']:
            group_data[f'{col}_min'] = group[col].min()
            group_data[f'{col}_max'] = group[col].max()
            group_data[f'{col}_mean'] = group[col].mean()
            group_data[f'{col}_std'] = group[col].std()

        group_data['Revenue (Millions $)_sum'] = group['Revenue (Millions $)'].sum()
        new_data.append(group_data)

    new_df = pd.DataFrame(new_data)
    new_df = new_df.sort_values(by=[group_by, 'Release Year'])

    lag_columns = ['Players (Millions)_mean', 'Peak Concurrent Players_mean', 'Metacritic Score_mean', 'Revenue (Millions $)_sum']
    for col in lag_columns:
        for i in range(1, 6):
            new_df[f'{col}_lag_{i}'] = new_df.groupby(group_by)[col].shift(i)

    new_df = new_df.reset_index(drop=True).fillna(-1000)
    new_df = new_df.rename(columns={'Revenue (Millions $)_sum': 'revenue_sum'})

    features = [col for col in new_df.columns if col not in [group_by, 'Release Year', 'revenue_sum']]
    X = new_df[features]
    y = new_df['revenue_sum']

    X_train = X[new_df['Release Year'] < 2020]
    X_test = X[new_df['Release Year'] >= 2020]
    y_train = y[new_df['Release Year'] < 2020]
    y_test = y[new_df['Release Year'] >= 2020]

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred[new_df['Release Year'] >= 2020]))
    mape = mean_absolute_percentage_error(y_test, y_pred[new_df['Release Year'] >= 2020])

    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**MAPE**: {mape:.2%}")

    target_data = new_df[new_df[group_by] == selected_value]
    years = target_data['Release Year']
    y_true = target_data['revenue_sum']
    y_pred_target = y_pred[new_df[group_by] == selected_value]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, y_true, label="Receita Real", marker='o')
    ax.plot(years[years >= 2020], y_pred_target[years >= 2020], label="Previsão", marker='x')
    ax.set_xlabel("Ano de Lançamento")
    ax.set_ylabel("Receita (Milhões $)")
    ax.set_title(f"Previsão de Receita para {selected_value}")
    ax.legend()
    st.pyplot(fig)

# Interface dinâmica com base na escolha
if menu == "Previsão por Plataforma":
    options = df['Platform'].dropna().unique()
    selected = st.selectbox("Escolha a Plataforma:", sorted(options))
    process_and_predict(df, 'Platform', selected)
else:
    options = df['Genre'].dropna().unique()
    selected = st.selectbox("Escolha o Gênero:", sorted(options))
    process_and_predict(df, 'Genre', selected)
