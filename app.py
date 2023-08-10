import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Constants
RISK_FREE_RATE = 0.02  # Assuming a risk-free rate of 2% for Sharpe/Treynor calculation


# Sample data
PORTFOLIOS = {
    'Tech Stocks': ['Apple', 'Microsoft', 'Google', 'Amazon', 'Facebook'],
    'Emerging Markets': ['Tencent', 'Alibaba', 'Samsung', 'Baidu', 'Xiaomi'],
    'Green Energy Assets': ['Tesla', 'NIO', 'Plug Power', 'First Solar', 'Enphase Energy']
}

DATE_RANGE = pd.date_range(start="2021-01-01", end="2021-12-31", freq='B')
ASSET_VALUES = np.random.rand(len(DATE_RANGE), 5) * 1000

NEWS_ITEMS = ["Positive news about Apple", "Google faces regulatory challenges", "Amazon grows in Europe", "Facebook under scrutiny", "Microsoft announces new partnership"]
SENTIMENTS = ['positive', 'neutral', 'negative']

def convert_to_cumulative_returns(values):
    returns = (values[1:] - values[:-1]) / values[:-1]
    cum_returns = np.cumsum(returns, axis=0)
    return np.vstack([[0]*values.shape[1], cum_returns]) * 100

def calculate_statistics(values, periods=[252, 252*3]):
    daily_returns = (values[1:] - values[:-1]) / values[:-1]
    statistics = {}

    for period in periods:
        returns = (values[-1] - values[-period]) / values[-period]
        std_dev = daily_returns[-period:].std()
        sharpe = (returns - RISK_FREE_RATE) / std_dev
        # For simplicity, using returns as proxy for beta in Treynor
        treynor = (returns - RISK_FREE_RATE) / returns
        statistics[period] = (returns, std_dev, sharpe, treynor)

    return statistics

# Streamlit app
st.title("Portfolio Insights Dashboard")

st.markdown("""
    <div style="color: red; font-size: 14px; font-style: italic;">
        All Figures, Including Stock Prices, Are Hypothetical And For Illustrative Purposes Only
    </div>
""", unsafe_allow_html=True)

portfolio_selected = st.sidebar.selectbox('Select a Portfolio:', list(PORTFOLIOS.keys()))
date_range = st.sidebar.date_input('Select date range:', [DATE_RANGE[0], DATE_RANGE[-1]])

st.sidebar.text("\n")
st.sidebar.header("Tabs:")
tab_selected = st.sidebar.radio("", ["Asset Overview", "Historical Performance", "Sentiment Analysis", "Predictive Forecasting","Portfolio Optimization","Meta-Labeling"])

if tab_selected == "Asset Overview":
    st.subheader("Asset Valuation")
    fig_valuation = px.bar(x=PORTFOLIOS[portfolio_selected], y=ASSET_VALUES[-1], labels={'x': 'Asset', 'y': 'Value'})
    st.plotly_chart(fig_valuation)

    st.subheader("Portfolio Distribution")
    fig_pie = px.pie(values=ASSET_VALUES[-1], names=PORTFOLIOS[portfolio_selected], title='Portfolio Distribution')
    st.plotly_chart(fig_pie)

    st.subheader("Asset Details")
    asset_df = pd.DataFrame({
        'Asset Name': PORTFOLIOS[portfolio_selected],
        'Current Value': ASSET_VALUES[-1],
        '% Change': (ASSET_VALUES[-1] - ASSET_VALUES[-2]) / ASSET_VALUES[-2] * 100,
        'Volume Traded': np.random.randint(10000, 1000000, size=(5,)),
        '52-week High': ASSET_VALUES.max(axis=0),
        '52-week Low': ASSET_VALUES.min(axis=0),
    })
    st.write(asset_df)

elif tab_selected == "Historical Performance":
    # Select the stock to visualize
    stock_to_display = st.selectbox('Select Stock for Historical Performance:', PORTFOLIOS[portfolio_selected], index=0)

    historical_cumulative_returns = convert_to_cumulative_returns(ASSET_VALUES)
    historical_df = pd.DataFrame(historical_cumulative_returns, index=DATE_RANGE, columns=PORTFOLIOS[portfolio_selected])

    fig_line = px.line(historical_df, x=historical_df.index, y=stock_to_display, title=f'Historical Performance for {stock_to_display} based on Cumulative Returns')
    # fig_line.update_layout(yaxis_tickformat='%')  # Format Y-axis as percentage

    st.plotly_chart(fig_line)


elif tab_selected == "Sentiment Analysis":
    st.subheader("Recent News")
    news_df = pd.DataFrame({
        'Headline': NEWS_ITEMS,
        'Sentiment': np.random.choice(SENTIMENTS, 5)
    })
    st.write(news_df)

    st.subheader("Sentiment Distribution")
    sentiment_counts = pd.DataFrame({'Sentiment': np.random.choice(SENTIMENTS, 5)}).value_counts().reset_index(name='counts')
    fig_sentiment = px.bar(sentiment_counts, x='Sentiment', y='counts', title='Sentiment Distribution')
    st.plotly_chart(fig_sentiment)

    st.subheader("Word Cloud")
    wordcloud_data = {
        'Apple': 50,
        'Google': 30,
        'Regulatory': 25,
        'Europe': 20,
        'Partnership': 15,
        'Growth': 10,
        'Batteries': 45,
        'Large Language Model': 40,
        'Dodge and Cox': 60,
        'Innovation': 35,
        'E-commerce': 28,
        'Blockchain': 33,
        'AI': 50,
        'Sustainability': 24,
        'Data Privacy': 32,
        'Fintech': 27,
        'Cloud Computing': 38,
        'Digital Transformation': 29,
        '5G': 31,
        'Machine Learning': 34
    }

    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)
   
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

elif tab_selected == "Predictive Forecasting":
    st.subheader("Predictive Forecasting for the Next Month based on Cumulative Returns")
    
    # Select the stock to visualize
    stock_to_display = st.selectbox('Select Stock for Forecasting:', PORTFOLIOS[portfolio_selected], index=0)

    # Generate a simple random walk forecast for demonstration purposes
    forecast_length = 20  # 20 business days ~ 1 month
    last_cumulative_return = convert_to_cumulative_returns(ASSET_VALUES)[-1]

    random_walk = np.random.randn(forecast_length, 5) * 10
    cumulative_random_walk = np.cumsum(random_walk, axis=0)
    forecast_cumulative_returns = last_cumulative_return + cumulative_random_walk

    # Append forecast to historical cumulative returns for visualization
    # Get the next 20 business days after the last DATE_RANGE entry
    historical_cumulative_returns = convert_to_cumulative_returns(ASSET_VALUES)
    next_business_days = pd.bdate_range(DATE_RANGE[-1] + timedelta(days=1), periods=forecast_length)
    all_dates = DATE_RANGE.union(next_business_days)
    all_cumulative_returns = np.vstack([historical_cumulative_returns, forecast_cumulative_returns])

    forecast_df = pd.DataFrame(all_cumulative_returns, index=all_dates, columns=PORTFOLIOS[portfolio_selected])

    fig_forecast = px.line(forecast_df, x=forecast_df.index, y=stock_to_display, title=f'Predictive Forecasting for {stock_to_display} based on Cumulative Returns')
    
    # Highlight the forecasted area
    fig_forecast.add_vrect(x0=DATE_RANGE[-1], x1=next_business_days[-1], fillcolor="LightSalmon", opacity=0.5, layer="below", line_width=0)

    st.plotly_chart(fig_forecast)

    st.write(f"Note: The shaded region represents the forecasted cumulative returns for the next month for {stock_to_display}. This forecasting is based on a simple random walk for demonstration purposes and is not suitable for actual financial predictions.")




elif tab_selected == "Portfolio Optimization":
    st.subheader("Hierarchical Risk Parity Portfolio Optimization")

    # Correcting the Hierarchical Clustering (HC)
    # Use correlation as distance measure
    corr_matrix = np.corrcoef(ASSET_VALUES, rowvar=False)
    distance_matrix = np.sqrt((1 - corr_matrix) / 2)
    Z = linkage(distance_matrix, 'ward')

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=PORTFOLIOS[portfolio_selected], ax=ax)
    st.pyplot(fig)

    st.write("The dendrogram above showcases the hierarchical clustering of assets, which can be used for risk parity-based portfolio optimization.")

elif tab_selected == "Meta-Labeling":
    st.subheader("Primary Model Predictions and Meta-Labeling Confidence")

    # Mock primary model predictions
    primary_predictions = np.random.choice(["Up", "Down"], size=5)
   
    # Mock meta-model confidence scores
    meta_confidences = np.random.rand(5)

    meta_df = pd.DataFrame({
        'Asset Name': PORTFOLIOS[portfolio_selected],
        'Primary Prediction': primary_predictions,
        'Meta-Model Confidence': meta_confidences
    })
    st.write(meta_df)

st.sidebar.text("\n\nNote: All data is fictional for demonstration purposes.")
