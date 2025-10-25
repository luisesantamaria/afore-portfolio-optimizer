# Afore Portfolio Optimizer

[![Launch App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://afore-portfolio-optimizer.streamlit.app/)

From regulatory complexity to your most efficient retirement portfolio.

This project is an interactive dashboard for analyzing and optimizing Mexican pension funds (Afores). It allows users to compare their current Siefore against a competitor, construct a custom asset portfolio, and calculate the optimal, efficient composition—all while adhering to the official CONSAR regulatory limits.

## Quick Access

- Dashboard: [Streamlit App](https://company-valuation.streamlit.app)  
- Notebook: [`notebook.ipynb`](notebook.ipynb)

## Highlights

* **Siefore Selection:** Choose your primary Afore/Siefore and select a competitor for a head-to-head comparison.
* **Custom Portfolio Constructor:** Build a universe of investable assets from various classes (e.g., Fixed Income, Local Equity, International Equity, Structured Notes).
* **Efficient Frontier Calculation:** Automatically calculates and visualizes the Markowitz efficient frontier based *only* on the assets you selected.
* **Optimal Portfolio:** Identifies the "best" portfolio on the frontier (e.g., max Sharpe Ratio) and displays its precise asset allocation (weights).
* **CONSAR Compliance:** The entire optimization process is constrained by the official, current investment regime (limits) published by CONSAR.
* **Contribution Simulation:** Model the long-term impact of adding voluntary contributions to your optimized portfolio.
* **Asset Recommendations:** Provides suggestions for new assets (from your selected universe) that could improve the portfolio's overall risk/return profile.
* **Key Metrics:** Displays the expected return, volatility, and Sharpe Ratio for the final optimized portfolio.
* **Data Export:** One-click download of the optimal portfolio weights as a CSV.

## Repository Structure

* `app.py` — The Streamlit dashboard application.
* `notebook.ipynb` — The data analysis and methodology notebook.
* `data/consar_limits.csv` — (Example) CSV holding the regulatory investment limits.
* `data/asset_universe.csv` — (Example) List of available assets for selection.
* `requirements.txt` — Project dependencies.
* `README.md` — This file.

## Data

* **Source:** Asset price history sourced from Yahoo Finance (via `yfinance`). Siefore performance and regulatory limits sourced from public CONSAR data.
* **Universe:** A curated list of selectable assets including local/international stocks, government/corporate bonds, and structured products.
* **Notes:** The dashboard runs optimizations in real-time. Asset recommendations are based on their marginal contribution to the portfolio's Sharpe Ratio.

## Methodology (Short)

* **Efficient Frontier:** Calculated using Modern Portfolio Theory (Markowitz). The model uses the expected returns (e.g., historical mean) and covariance matrix of the user-selected assets.
* **Optimization Goal:** The "optimal" portfolio is found by maximizing the Sharpe Ratio (return per unit of risk).
* **Constraints:** The optimization is solved using a Sequential Least Squares Programming (SLSQP) method, subject to two key constraints:
    1.  All asset weights must sum to 1 (`sum(weights) = 1`).
    2.  The sum of weights for each asset class (Equity, Fixed Income, etc.) must be *less than or equal to* the maximum percentage allowed by the official CONSAR investment regime.
