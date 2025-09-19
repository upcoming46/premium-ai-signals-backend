# Premium AI Binary Signals Backend

Advanced AI-powered binary options signal generation system with 85-95% accuracy target.

## Features

- **Advanced ML Models**: Gradient Boosting, Random Forest, and Calibrated Classifiers
- **Multi-timeframe Confluence**: Analyzes 1m, 5m, and 15m timeframes
- **News Sentiment Analysis**: Real-time market sentiment integration
- **Pattern Recognition**: Volume-confirmed candlestick patterns
- **Risk Management**: Kelly Criterion-based position sizing
- **Session Timing**: Optimized for London/NY trading sessions
- **Performance Tracking**: Historical win rate and pattern analysis
- **Advanced Telegram Notifications**: Detailed signal analysis

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Render Deployment

1. **Create Render Account**: Sign up at render.com
2. **New Web Service**: Connect this GitHub repository
3. **Configure Build**:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

4. **Environment Variables**:
```
OANDA_TOKEN=your_oanda_token
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
NEWS_KEY=your_news_api_key
POLYGON_KEY=your_polygon_key
FINNHUB_KEY=your_finnhub_key
FRED_KEY=your_fred_key
DATABASE_URL=postgresql://user:pass@host/db
REDIS_URL=redis://host:port/0
SECRET_KEY=your_secret_key
```

## API Endpoints

- `GET /health` - Health check with feature status
- `GET /signals/{asset}` - Generate premium signal for asset
- `GET /performance/{asset}` - Get performance statistics
- `GET /news-sentiment/{asset}` - Get news sentiment analysis

## Signal Generation Process

1. **Session Filter**: Validates optimal trading hours (London/NY overlap)
2. **Data Fetch**: Retrieves real-time price data from OANDA
3. **Confluence Scoring**: Multi-timeframe technical analysis
4. **Pattern Detection**: Volume-confirmed candlestick patterns
5. **Sentiment Analysis**: News-based market sentiment
6. **ML Prediction**: Calibrated classifier confidence scoring
7. **Risk Sizing**: Kelly Criterion position sizing
8. **Telegram Notification**: Advanced formatted alerts

## Supported Assets

**Spot**: EURUSD, GBPUSD, AUDUSD, USDJPY, EURGBP, EURJPY, GBPJPY, AUDJPY, USDCHF, NZDUSD
**OTC**: All spot assets with -OTC suffix

## Performance Targets

- **Accuracy**: 85-95% historical performance
- **Confidence Range**: 75-95% per signal
- **Risk Management**: 0.5-5% position sizing
- **Session Boost**: +11% during optimal hours

## License

MIT License - See LICENSE file for details

