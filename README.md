# Financial Analyst - Stock Analysis Dashboard

A comprehensive stock analysis tool that combines **fundamental** and **technical** analysis to provide holistic buy/sell/hold recommendations.

## Features

### Fundamental Analysis
- **Valuation**: P/E, P/B, P/S, PEG, EV/EBITDA
- **Profitability**: ROE, ROA, Gross/Operating/Net margins
- **Growth**: Revenue, earnings, and EPS growth rates
- **Financial Health**: Debt-to-equity, current ratio, quick ratio
- **Dividends**: Yield, payout ratio

### Technical Analysis
- **Trend**: SMA (20/50/200), EMA, MACD
- **Momentum**: RSI, Stochastic Oscillator
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume SMA
- **Support/Resistance**: Pivot points

### Holistic Recommendation
Weighted scoring system combining fundamental (50%) and technical (50%) signals into a unified recommendation: **Strong Buy / Buy / Hold / Sell / Strong Sell**

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then enter any stock ticker (e.g., AAPL, MSFT, GOOGL) to get a full analysis.
