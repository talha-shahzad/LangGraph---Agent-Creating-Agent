import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.spatial.distance import cosine

class ToolRegistry:
    """Enhanced SQLite-backed tool registry with embeddings and caching."""
    
    def __init__(self, connection_string: str):
        self.db_path = connection_string.replace("sqlite:///", "", 1)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with enhanced schema."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Tools table with embeddings and metrics
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                code TEXT NOT NULL,
                parameters TEXT,
                embedding TEXT,  -- JSON string of vector
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                approved_by TEXT,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_runtime FLOAT DEFAULT 0,
                last_used DATETIME,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Result cache table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS result_cache (
                cache_key TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME
            )
        """)

        # Tool executions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tool_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT,
                input_params TEXT,
                output TEXT,
                success INTEGER,
                runtime FLOAT,
                error_message TEXT,
                executed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Pending approvals table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pending_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT UNIQUE,
                session_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # --- Migrations ---
        
        # Tools table migration
        cur.execute("PRAGMA table_info(tools)")
        tools_cols = [column[1] for column in cur.fetchall()]
        tools_migrations = [
            ("embedding", "TEXT"),
            ("success_count", "INTEGER DEFAULT 0"),
            ("failure_count", "INTEGER DEFAULT 0"),
            ("avg_runtime", "FLOAT DEFAULT 0")
        ]
        for col_name, col_type in tools_migrations:
            if col_name not in tools_cols:
                logging.info(f"Migrating: Adding column {col_name} to tools table")
                cur.execute(f"ALTER TABLE tools ADD COLUMN {col_name} {col_type}")

        # Tool executions table migration
        cur.execute("PRAGMA table_info(tool_executions)")
        exec_cols = [column[1] for column in cur.fetchall()]
        if "runtime" not in exec_cols:
            logging.info("Migrating: Adding column runtime to tool_executions table")
            cur.execute("ALTER TABLE tool_executions ADD COLUMN runtime FLOAT")

        conn.commit()
        conn.close()

    def save_tool(self, name: str, description: str, code: str, parameters: Dict, 
                  embedding: List[float], approved_by: str, status: str = 'pending') -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO tools (name, description, code, parameters, embedding, approved_by, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (name) DO UPDATE SET
                    description = EXCLUDED.description,
                    code = EXCLUDED.code,
                    parameters = EXCLUDED.parameters,
                    embedding = EXCLUDED.embedding,
                    approved_by = EXCLUDED.approved_by,
                    status = EXCLUDED.status
            """, (name, description, code, json.dumps(parameters), json.dumps(embedding), approved_by, status))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logging.error(f"Error saving tool {name}: {e}")
            return False

    def get_tool(self, name: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM tools WHERE name = ? AND status = 'active'", (name,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None

    def search_tools(self, query_embedding: List[float], threshold: float = 0.5) -> List[Dict]:
        """Semantic search using cosine similarity."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT name, description, code, parameters, embedding, usage_count, success_count FROM tools WHERE status = 'active'")
        rows = cur.fetchall()
        conn.close()

        results = []
        for row in rows:
            if not row['embedding']: continue
            tool_emb = json.loads(row['embedding'])
            similarity = float(1 - cosine(query_embedding, tool_emb))
            if similarity >= threshold:
                tool_data = dict(row)
                tool_data['similarity'] = similarity
                results.append(tool_data)
        
        # Sort by similarity, then usage frequency
        results.sort(key=lambda x: (x['similarity'], x['usage_count']), reverse=True)
        return results

    def get_cached_result(self, cache_key: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT result FROM result_cache 
            WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        """, (cache_key,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None

    def set_cache_result(self, cache_key: str, result: str, ttl_hours: int = 24):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        expires_at = (datetime.now() + timedelta(hours=ttl_hours)).strftime('%Y-%m-%d %H:%M:%S')
        cur.execute("""
            INSERT INTO result_cache (cache_key, result, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT (cache_key) DO UPDATE SET
                result = EXCLUDED.result,
                expires_at = EXCLUDED.expires_at,
                created_at = CURRENT_TIMESTAMP
        """, (cache_key, result, expires_at))
        conn.commit()
        conn.close()

    def log_execution(self, tool_name: str, input_params: Dict, output: str, 
                      success: bool, runtime: float, error_message: str = None):
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Log detail
            cur.execute("""
                INSERT INTO tool_executions (tool_name, input_params, output, success, runtime, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tool_name, json.dumps(input_params), output, 1 if success else 0, runtime, error_message))
            
            # Update metrics
            success_inc = 1 if success else 0
            fail_inc = 0 if success else 1
            
            cur.execute("""
                UPDATE tools SET 
                    usage_count = usage_count + 1,
                    success_count = success_count + ?,
                    failure_count = failure_count + ?,
                    avg_runtime = (avg_runtime * usage_count + ?) / (usage_count + 1),
                    last_used = CURRENT_TIMESTAMP
                WHERE name = ?
            """, (success_inc, fail_inc, runtime, tool_name))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error logging execution: {e}")

    def get_all_tools(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM tools ORDER BY usage_count DESC")
        rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def approve_tool(self, name: str, approved_by: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE tools SET status = 'active', approved_by = ? WHERE name = ?", (approved_by, name))
        cur.execute("DELETE FROM pending_approvals WHERE tool_name = ?", (name,))
        conn.commit()
        conn.close()
        return True

    def reject_tool(self, name: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE tools SET status = 'rejected' WHERE name = ?", (name,))
        cur.execute("DELETE FROM pending_approvals WHERE tool_name = ?", (name,))
        conn.commit()
        conn.close()
        return True

    def add_pending_approval(self, tool_name: str, session_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO pending_approvals (tool_name, session_id) VALUES (?, ?) ON CONFLICT DO NOTHING", (tool_name, session_id))
        conn.commit()
        conn.close()
        return True

    def get_pending_approvals(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT t.*, pa.session_id, pa.created_at as pending_since
            FROM tools t
            JOIN pending_approvals pa ON t.name = pa.tool_name
            WHERE t.status = 'pending'
        """)
        rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]
