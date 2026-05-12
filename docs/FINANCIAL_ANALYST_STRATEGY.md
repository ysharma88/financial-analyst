# Financial Analyst Strategy

Core scripts now live in this tool (not CPQ):

- `scripts/financial_strategy.py`
- `scripts/execution_algorithms.py`
- `scripts/data_feed.py`
- `scripts/financial_report.py`
- `scripts/cpq_financial_bridge.py`
- `scripts/run_financial_analyst.py`
- `scripts/telegram_market_poller.py`

## Run Locally

```bash
cd "/Users/yogesh.sharma/Cursor Project/financial_analyst"
python3 scripts/run_financial_analyst.py AAPL --data-source yahoo --period 3mo
```

## Telegram Polling

```bash
export TELEGRAM_BOT_TOKEN="<bot_token>"
export TELEGRAM_CHAT_ID="<chat_id>"
python3 scripts/telegram_market_poller.py --symbols AAPL,MSFT,NVDA --poll-seconds 300
```

## Use Your Tracker Portfolio

```bash
python3 scripts/telegram_market_poller.py --tracker-file "/path/to/tracker.json" --poll-seconds 300
```

## UI-Driven Tracker List

- In the Streamlit sidebar, use **Portfolio Tracker (UI)** to add/remove symbols.
- Click **Save Tracker List** to persist.
- The poller automatically reads `config/ui_tracked_stocks.json` when `--tracker-file` is not provided.

## Dynamic Alert Controls

- In the Streamlit sidebar, use **Alert Settings (UI)** and click **Save Alert Settings**.
- This writes `config/alert_settings.json` with:
  - `poll_seconds`
  - `price_jump_threshold_pct`
  - `atr_spike_threshold_pct`
  - `momentum_spike_abs_pct`
- The poller reloads this file every cycle, so changes apply without code edits.
- Run poller with advanced threshold evaluation enabled:

```bash
python3 scripts/telegram_market_poller.py --enable-advanced-metrics --poll-seconds 300
```

## Telegram Credentials in UI

- In the Streamlit sidebar, use **Telegram Bot Credentials**.
- Enter:
  - Bot Token
  - Chat ID
- Click **Save Telegram Credentials**.
- Use **Send Test Message** to verify bot delivery from the UI immediately.
- Credentials are stored in:
  - `config/telegram_credentials.json`
- File permissions are set to owner-only when supported (`chmod 600`), and the file is gitignored.
- Poller auto-loads these credentials (environment variables still override file values).

