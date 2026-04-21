# Crypto Pulse

Crypto Pulse is a compact crypto signal project built around one idea:
predict whether the next move is likely to be stronger or weaker than the
previous move, instead of naively guessing whether price will go up or down.

The project includes:

- feature engineering from hourly Binance OHLCV data
- an LSTM + attention classifier for 3-day and 7-day horizons
- walk-forward validation with threshold tuning
- a backtest with transaction-cost stress testing
- a FastAPI service
- a Streamlit dashboard

## Project Layout

- `features.py`: data download, caching, feature engineering, sequence creation
- `model.py`: custom attention layer and model builder
- `train.py`: walk-forward training and metrics export
- `backtest.py`: threshold-based strategy simulation
- `core.py`: shared prediction engine used by the app and API
- `api.py`: FastAPI service
- `app.py`: Streamlit dashboard
- `config.py`: central configuration

This split is intentional. The main runtime surfaces (`app.py`, `api.py`) stay
thin, while the model and data pipeline stay reusable.

## How The Signal Works

For each horizon, the target asks whether the upcoming return will outperform
the immediately preceding return over the same horizon.

That means:

- `UP` does not strictly mean "price will rise"
- `DOWN` does not strictly mean "price will fall"
- the signal is better interpreted as stronger vs weaker forward momentum

## Current Model Setup

- Assets: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`
- Interval: `1h`
- Sequence length: `48` timesteps
- Horizons: `3d`, `7d`
- Model: LSTM + temporal attention + layer normalization

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Run the backtest:

```bash
python backtest.py
```

Start the API:

```bash
python api.py
```

Start the dashboard:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## API

Environment variables:

- `HOST`: bind address, defaults to `0.0.0.0`
- `PORT`: service port, defaults to `8000`
- `ALLOWED_ORIGINS`: comma-separated CORS origins, defaults to `*`

Health endpoints:

- `GET /healthz`: liveness
- `GET /readyz`: model readiness

Main endpoints:

- `GET /predict/{symbol}`
- `GET /metrics`
- `GET /backtest`

## Docker

**Important**: If deploying to environments where Binance API is geo-blocked (returns 451 errors), you must pre-cache data locally before building:

```bash
# Pre-cache data (run on a machine with Binance access)
python cache_data.py

# Then build the Docker image
docker build -f Dockerfile.api -t crypto-pulse-api .
docker run -p 8000:8000 crypto-pulse-api
```

Environment variables for deployment:

- `ALLOW_STALE_CACHE=true`: Allow using cached data even if older than 12 hours
- `HTTP_PROXY` or `HTTPS_PROXY`: Use a proxy server to access Binance API
- `BINANCE_BASE_URLS`: Comma-separated list of Binance API endpoints to try

Build and run the API container:

Build and run the Streamlit container:

```bash
docker build -f Dockerfile.streamlit -t crypto-pulse-ui .
docker run -p 8501:8501 crypto-pulse-ui
```

**Note**: The same geo-blocking considerations apply to Streamlit deployment as the API.

## Notes

- Live predictions require network access to Binance.
- If you encounter "451 Client Error" (geo-blocking), use a proxy server or pre-cache data locally.
- Trained artifacts are expected in `models/`.
- Backtest output is written to `models/backtest.json`.
- Walk-forward metrics are written to `models/metrics.json`.

## Disclaimer

This project is for educational and portfolio purposes only. It is not
financial advice.

## Streamlit Cloud
### Direct Deployment

You can deploy directly to Streamlit Cloud:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Select `app.py` as the main file
4. Set the secret `ALLOW_STALE_CACHE` to `true`
5. Deploy!

**Note**: The cached data and trained model are already committed to the repository.

For Streamlit Cloud deployment:

1. Pre-cache data locally:
   ```bash
   python cache_data.py
   ```

2. Commit the cached data:
   ```bash
   git add data/cache/
   git commit -m "Add cached Binance data for deployment"
   ```

3. Deploy to Streamlit Cloud with these secrets:
   - `ALLOW_STALE_CACHE`: `true`
