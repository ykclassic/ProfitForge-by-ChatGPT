-- CORE
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name TEXT,
    version TEXT,
    algorithm TEXT,
    trained_at TIMESTAMPTZ,
    data_window INTEGER
);

-- SIGNALS
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ,
    symbol TEXT,
    direction TEXT,
    confidence REAL,
    status TEXT DEFAULT 'NEW'
);

-- MODEL METRICS
CREATE TABLE IF NOT EXISTS model_metrics (
    model_id INTEGER,
    metric TEXT,
    value REAL,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- WALK-FORWARD
CREATE TABLE IF NOT EXISTS walkforward_results (
    model_id INTEGER,
    train_start TIMESTAMPTZ,
    train_end TIMESTAMPTZ,
    test_start TIMESTAMPTZ,
    test_end TIMESTAMPTZ,
    accuracy REAL,
    precision REAL,
    recall REAL,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- REGIME
CREATE TABLE IF NOT EXISTS market_regimes (
    timestamp TIMESTAMPTZ PRIMARY KEY,
    symbol TEXT,
    regime INTEGER,
    probability REAL
);

-- BET SIZING
CREATE TABLE IF NOT EXISTS bet_sizing (
    signal_id INTEGER,
    probability REAL,
    kelly REAL,
    fraction REAL,
    capital_used REAL,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- CONCEPT DRIFT
CREATE TABLE IF NOT EXISTS concept_drift (
    timestamp TIMESTAMPTZ,
    model_id INTEGER,
    metric TEXT,
    drift_detected BOOLEAN,
    value REAL
);
