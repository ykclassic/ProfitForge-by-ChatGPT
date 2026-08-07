import sqlite3
from typing import List, Dict, Optional

class TradingDatabaseHandler:
    def __init__(self, db_path: str = "trading_engine.db"):
        self.db_path = db_path
        self._initialize_optimizations()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Creates a secure, optimized connection.
        Enforces foreign keys and enables WAL mode for high-frequency concurrent operations.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _initialize_optimizations(self) -> None:
        """Applies necessary indexes to prevent full table scans during frequent queries."""
        index_queries = """
        CREATE INDEX IF NOT EXISTS idx_signals_symbol_id ON trading_signals (symbol, id DESC);
        CREATE INDEX IF NOT EXISTS idx_walkforward_accuracy ON walkforward_results (accuracy DESC);
        """
        with self._get_connection() as conn:
            conn.executescript(index_queries)

    def get_latest_signal_status(self, terminal_symbol: str) -> Optional[Dict]:
        """
        Fetches the execution state of the most recent signal for a specific terminal asset.
        Uses exact terminal symbols (no pair mapping).
        """
        query = """
            SELECT id, timestamp, direction, entry, sl, tp, confidence, status
            FROM trading_signals
            WHERE symbol = ?
            ORDER BY id DESC
            LIMIT 1;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (terminal_symbol,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None

    def get_models_by_accuracy(self, min_accuracy: float) -> List[Dict]:
        """
        Filters production-ready models based on out-of-sample walk-forward accuracy.
        """
        query = """
            SELECT 
                m.id AS model_id, 
                m.name, 
                m.version, 
                w.accuracy, 
                w.precision, 
                w.test_start, 
                w.test_end
            FROM models m
            JOIN walkforward_results w ON m.id = w.model_id
            WHERE w.accuracy >= ?
            ORDER BY w.accuracy DESC;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (min_accuracy,))
            return [dict(row) for row in cursor.fetchall()]

# --- Usage Execution ---
if __name__ == "__main__":
    db = TradingDatabaseHandler()
    
    # Example 1: Fetching latest status using an exact terminal symbol
    latest_signal = db.get_latest_signal_status("BTCUSDT.P")
    if latest_signal:
        print(f"Latest Signal Status: {latest_signal['status']} for trade ID {latest_signal['id']}")
        
    # Example 2: Filtering for models with >= 65% out-of-sample accuracy
    viable_models = db.get_models_by_accuracy(0.65)
    for model in viable_models:
        print(f"Model {model['name']} v{model['version']} passed with {model['accuracy']} accuracy.")
