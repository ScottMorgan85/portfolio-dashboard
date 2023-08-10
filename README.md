# Portfolio Insights Dashboard

Streamlit Cloud deployment: https://portfolio-dashboard-example-scott-morgan.streamlit.app/

## Overview

The **Portfolio Insights Dashboard** is a data-driven application aimed at providing insightful analytics for a set of hypothetical portfolios. Through a modern, interactive interface, users can explore different facets of the portfolios, including asset valuations, historical performance, sentiment analysis derived from news items, and predictive forecasting. This application serves as an illustrative example of how to integrate data visualization, financial calculations, and sentiment analysis in a user-friendly dashboard.

### Key Features

1. **Asset Overview**: Get a quick glimpse of the valuation of assets in the selected portfolio. Explore the portfolio's distribution and delve into detailed metrics, such as volume traded and 52-week high/lows.

2. **Historical Performance**: Visualize the cumulative returns of assets over time. Key financial statistics, including 1-year and 3-year returns, standard deviations, Sharpe, and Treynor ratios, are also provided.

3. **Sentiment Analysis**: Stay updated with the latest news items related to the assets in the portfolio. A sentiment distribution chart and word cloud give insights into the prevailing mood surrounding the assets.

4. **Predictive Forecasting**: Witness the power of forecasting as the dashboard provides a month-long prediction for the assets based on a simple random walk model.

5. **Portfolio Optimization**: Analyze the hierarchical risk parity portfolio optimization through a dendrogram that showcases the hierarchical clustering of assets.

6. **Meta-Labeling**: Explore the primary model predictions for each asset along with confidence scores derived from a hypothetical meta-model.

### Technical Stack

- **Streamlit**: Serves as the backbone of the application, enabling rapid development and deployment of the dashboard.
  
- **Pandas**: Used for efficient data manipulation and calculations.
  
- **Plotly Express**: Powers the interactive visualizations within the dashboard.
  
- **Scipy & Matplotlib**: Integral for hierarchical clustering and dendrogram visualization in the portfolio optimization section.

### Usage

To get started, simply clone the repository and set up a Python environment with the required libraries installed. Run the Streamlit application and navigate through the sidebar to select different portfolios and dashboard views.

### Note

_All data presented in this application is fictional and for demonstration purposes only. It's essential to cross-check any findings or insights with real-world data and expert opinions before making investment decisions._
