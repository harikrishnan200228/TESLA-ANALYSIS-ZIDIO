# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Tesla Stock Analysis", layout="wide")

st.title("📈 Tesla Stock Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload Tesla.csv", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Data Cleaning
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)

    # Feature Engineering
    if 'Close' in df.columns:
        df['Daily Return'] = df['Close'].pct_change()
        df['Moving_Avg_20'] = df['Close'].rolling(window=20).mean()
        df['Moving_Avg_50'] = df['Close'].rolling(window=50).mean()
        df['Volatility_20'] = df['Daily Return'].rolling(window=20).std()

    # Overview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Price Trend
    if 'Close' in df.columns:
        st.subheader("Tesla Stock Price Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

    # Daily Returns Distribution
    if 'Daily Return' in df.columns:
        st.subheader("Distribution of Daily Returns")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['Daily Return'].dropna(), bins=50, kde=True, color='purple', ax=ax)
        st.pyplot(fig)

    # Volume Over Time
    if 'Volume' in df.columns:
        st.subheader("Tesla Trading Volume Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'], df['Volume'], label='Volume', color='orange')
        ax.legend()
        st.pyplot(fig)

    # Close Price with Moving Averages
    if all(col in df.columns for col in ['Close', 'Moving_Avg_20', 'Moving_Avg_50']):
        st.subheader("Close Price with Moving Averages")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        ax.plot(df['Date'], df['Moving_Avg_20'], label='20-Day MA', color='red')
        ax.plot(df['Date'], df['Moving_Avg_50'], label='50-Day MA', color='green')
        ax.legend()
        st.pyplot(fig)

    # Insights
    st.subheader("Key Insights")
    if 'Close' in df.columns:
        st.write(f"**Highest Closing Price:** {df['Close'].max():.2f} USD")
        st.write(f"**Lowest Closing Price:** {df['Close'].min():.2f} USD")
    if 'Daily Return' in df.columns:
        st.write(f"**Max Daily Return:** {df['Daily Return'].max():.4f}")
        st.write(f"**Min Daily Return:** {df['Daily Return'].min():.4f}")
    if 'Volume' in df.columns:
        st.write(f"**Highest Volume:** {df['Volume'].max()}")
        st.write(f"**Lowest Volume:** {df['Volume'].min()}")

else:
    st.info("Please upload Tesla.csv to start the analysis.")
