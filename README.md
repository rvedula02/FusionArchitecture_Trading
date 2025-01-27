# Fusion Stock Predictor

A deep learning framework for high-frequency stock price prediction using my attempt at a fusion architecture that combine intra-stock temporal modeling with market-guided feature selection.

## Overview

This project implements a parallel fusion architecture for stock price forecasting that:

- Models momentary and cross-time stock correlations
- Guides feature selection using market information
- Utilizes patch-based processing for improved efficiency
- Combines multiple prediction heads for returns and directional forecasting

The architecture is inspired by two papers: MASTER framework and incorporates elements from aLLM4TS for time series representation learning.

## Architecture

The model consists of three main components:

1. **Intra-Stock Attention**: Models temporal dependencies within individual stocks using transformer-based attention mechanisms

2. **Market Gating**: Dynamically weights features based on market conditions using a gating mechanism

3. **Cross-Time Attention**: Captures relationships across different time periods
