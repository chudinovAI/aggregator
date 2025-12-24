## Setup

1) Install deps (Python 3.11+):
```
pip install -e .
```
2) Export API credentials via environment variables **or** point `AGG_SECRETS_PATH`
   to an external JSON file (kept outside the repo). Example secrets file:
   ```json
   {
     "REDDIT_CLIENT_ID": "xxx",
     "REDDIT_CLIENT_SECRET": "yyy",
     "YOUTUBE_API_KEY": "zzz"
   }
   ```
   Then run:
   ```
   export AGG_SECRETS_PATH=$HOME/.config/aggregator/secrets.json
   ```
   Keys found in the environment always override the file.

## Train classifiers

Classic TF-IDF + logistic regression:
```
python bin/train_classifier.py
```
- Unicode-aware normalization keeps characters from any language (Latin, Cyrillic,
  etc.), so you can mix multilingual examples within the CSV dataset.

Transformer (multilingual DistilBERT, config-driven paths):
```
python bin/train_transformer.py
```
Dataset defaults to `data/news_classification_dataset.csv` (override via `AGG_TRANSFORMER_DATASET_PATH`).

## Run aggregator

Generate HTML/JSON/MD/RSS with async collectors, ML filtering, dedup, sentiment:
```
python bin/news_aggregator.py
```
Reports go to `docs/` by default; logs are JSON-formatted.***

### Rate limiting & retries

- Configure per-host throttling via:
  ```
  AGG_API_RATE_LIMIT_PER_HOST=90
  AGG_API_RATE_LIMIT_WINDOW_SECONDS=60
  ```
  Collectors share a global async rate limiter layered on top of exponential backoff (`api_retry_base_delay`, `api_retries`).
- Additional jitter between calls is controlled with `AGG_API_REQUEST_DELAY_SECONDS`.
- Per-source overrides (e.g., stricter YouTube quota) can be passed as JSON:
  ```
  AGG_API_RATE_LIMIT_PROFILES='{"youtube": [30, 60], "hackernews": [45, 60]}'
  ```
- All outbound links (collectors + HTML report) are validated against
  `AGG_ALLOWED_URL_SCHEMES` (defaults to `["http","https"]`) to block injected
  `javascript:` or other unsafe URLs.
  ```
  AGG_ALLOWED_URL_SCHEMES='["http","https"]'
  ```

### Storage & analytics

- Default storage is SQLite (`data/aggregator.db`). Switch to Postgres with:
  ```
  AGG_STORAGE_BACKEND=postgres
  AGG_STORAGE_PG_DSN=postgresql://user:pass@host:5432/aggregator
  ```
- Enable Redis caching for trend summaries:
  ```
  AGG_REDIS_ENABLED=true
  AGG_REDIS_URL=redis://localhost:6379/0
  AGG_REDIS_CACHE_TTL_SECONDS=900
  ```
- Reports now embed storage-driven analytics:
  - Source + sentiment breakdown sourced from persisted history (`AGG_STORAGE_SUMMARY_DAYS`, default 30).
  - Inline Chart.js charts for post volume and sentiment trends.

### Secrets management

- `.env` files inside the repo are no longer used. Provide credentials via OS
  environment or a secrets file referenced by `AGG_SECRETS_PATH`.
- Secrets file format is plain JSON (`{"KEY": "value"}`) stored outside the repo,
  ideally in an encrypted volume or secret manager mount.
- `load_runtime_secrets()` leaves pre-existing environment variables untouched, so
  container orchestrators (Kubernetes, ECS, etc.) remain the source of truth.

### Feedback loop & quality metrics

- After every run, top posts plus a configurable negative sample (`AGG_FEEDBACK_NEGATIVE_SAMPLE_SIZE`) are exported to `data/feedback_queue.jsonl`.
- Analysts can append labeled events as JSON lines to `data/feedback_events.jsonl`, e.g.:
  ```json
  {"post_id": "...", "label": "relevant"}
  ```
- The aggregator ingests events, appends them to `data/feedback_dataset.jsonl`, and surfaces precision/recall metrics directly in each report.
- Tuning knobs:
  - Disable the loop with `AGG_FEEDBACK_ENABLED=false`.
  - Control batch size for retraining readiness via `AGG_FEEDBACK_MIN_TRAINING_BATCH`.
  - Auto fine-tune the transformer as soon as enough feedback arrives:
    ```
    AGG_FEEDBACK_AUTOTRAIN_ENABLED=true
    AGG_TRANSFORMER_FEEDBACK_WEIGHT=2.0
    ```
    This retrains `bin/train_transformer.py` logic in-process using the base CSV + feedback JSONL mix (weighted towards human labels).
  - Override transformer hyperparameters if needed:
    ```
    AGG_TRANSFORMER_MODEL_NAME=distilbert-base-multilingual-cased
    AGG_TRANSFORMER_MAX_LENGTH=256
    AGG_TRANSFORMER_BATCH_SIZE=8
    AGG_TRANSFORMER_NUM_EPOCHS=3
    AGG_TRANSFORMER_LEARNING_RATE=2e-5
    ```
- Training metrics (loss/accuracy/F1, dataset sizes, hyperparams) are appended to
  `models/news_transformer/training_metrics.jsonl` every time the transformer
  trains (either via CLI or the auto fine-tune loop).

### Optional: digest email

Set SMTP env vars (prefixed with `AGG_` when using `.env`):
```
AGG_EMAIL_NOTIFICATIONS_ENABLED=true
AGG_EMAIL_SMTP_HOST=smtp.example.com
AGG_EMAIL_SMTP_PORT=587
AGG_EMAIL_SMTP_USERNAME=bot@example.com
AGG_EMAIL_SMTP_PASSWORD=your_password
AGG_EMAIL_SENDER=bot@example.com
AGG_EMAIL_RECIPIENTS='user1@example.com,user2@example.com'
```
The aggregator will send a plain-text summary once reports are generated.

### Optional: personalization

Enable personalization and create `data/user_profiles.json`:
```
AGG_PERSONALIZATION_ENABLED=true
AGG_USER_PROFILES_PATH=data/user_profiles.json
```

Example profile file:
```json
[
  {
    "name": "Alice",
    "email": "alice@example.com",
    "interests": ["python", "cuda"],
    "excluded_sources": ["ted_youtube"],
    "digest_limit": 8
  },
  {
    "name": "Bob",
    "email": "bob@example.com",
    "interests": ["hackernews", "open source"],
    "digest_limit": 5
  }
]
```
Each profile gets a personalized email (requires SMTP settings above).
