"""Process-safe, conservatively reserved step budgets.

Reserve blocks before stepping. Clean close releases unused reservations. A crash
may overcharge at most one block per worker; it cannot silently undercount steps.
Each worker must own its own lease. SQLite serializes reservations across workers.
"""

import sqlite3
import uuid
from pathlib import Path

import gymnasium as gym


class BudgetExhausted(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, limit=None):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, cap INTEGER)")
            db.execute("""CREATE TABLE IF NOT EXISTS leases (
                id TEXT PRIMARY KEY, label TEXT, charged INTEGER, used INTEGER, closed INTEGER)""")
            existing = db.execute("SELECT cap FROM budget WHERE id=1").fetchone()
            if existing is None:
                if limit is None or limit <= 0:
                    raise ValueError("A new ledger requires a positive step limit")
                db.execute("INSERT INTO budget VALUES (1, ?)", (limit,))
            elif limit is not None and limit != existing[0]:
                raise ValueError("Existing budget cap differs; use a new ledger for a new campaign")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def status(self):
        with self.connect() as db:
            cap = db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
            charged, used = db.execute(
                "SELECT COALESCE(SUM(charged),0),COALESCE(SUM(used),0) FROM leases"
            ).fetchone()
            leases = db.execute("SELECT label,charged,used,closed FROM leases").fetchall()
        return {
            "cap": cap,
            "charged_steps": charged,
            "confirmed_steps": used,
            "remaining_steps": cap - charged,
            "leases": [dict(zip(("label", "charged", "used", "closed"), row)) for row in leases],
        }

    def lease(self, label, block=256):
        if block < 1:
            raise ValueError("Reservation block must be positive")
        return Lease(self, label, block)


class Lease:
    def __init__(self, ledger, label, block):
        self.ledger, self.block = ledger, block
        self.id = uuid.uuid4().hex
        self.used = self.charged = 0
        self.closed = False
        with ledger.connect() as db:
            db.execute("INSERT INTO leases VALUES (?,?,0,0,0)", (self.id, label))

    def consume(self):
        if self.closed:
            raise RuntimeError("Lease already closed")
        if self.used == self.charged:
            with self.ledger.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                cap = db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
                total = db.execute("SELECT COALESCE(SUM(charged),0) FROM leases").fetchone()[0]
                amount = min(self.block, cap - total)
                if amount <= 0:
                    raise BudgetExhausted("Step budget exhausted; no environment action sent")
                db.execute(
                    "UPDATE leases SET charged=charged+?,used=? WHERE id=?",
                    (amount, self.used, self.id),
                )
            self.charged += amount
        self.used += 1

    def close(self):
        if not self.closed:
            with self.ledger.connect() as db:
                db.execute(
                    "UPDATE leases SET charged=?,used=?,closed=1 WHERE id=?",
                    (self.used, self.used, self.id),
                )
            self.charged = self.used
            self.closed = True


class StepMeter(gym.Wrapper):
    def __init__(self, env, lease):
        super().__init__(env)
        self.lease = lease

    def step(self, action):
        self.lease.consume()
        return self.env.step(action)

    def close(self):
        try:
            self.env.close()
        finally:
            self.lease.close()
