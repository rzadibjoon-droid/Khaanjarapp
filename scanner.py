"""
═══════════════════════════════════════════════════════════════════════════════
🔥 KHANJAR SUPREME V5.0 — SCANNER MODULE (Android Compatible)
═══════════════════════════════════════════════════════════════════════════════
"""

import requests
import numpy as np
import concurrent.futures
import warnings
import json
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

BASE = 'https://api.binance.com'
HEADERS = {'User-Agent': 'KHANJAR-SUPREME/5.0'}

# 🛡️ SAFETY
MAX_TRADES_PER_DAY = 3
MAX_DAILY_DRAWDOWN_PCT = -4.0
MIN_VOLUME_24H_USD = 5_000_000
MAX_SPREAD_PCT = 0.15
MAX_CONSECUTIVE_LOSSES = 3

# 💰 ACCOUNT
ACCOUNT_BALANCE_USD = 30
RISK_PER_TRADE_PCT = 2.0
MAX_POSITION_PCT = 15.0
MIN_POSITION_USD = 5

# 📊 LEVERAGE
MAX_SAFE_LEVERAGE = 5

# 📁 STATE (Android-compatible path)
try:
    # Try Android external storage first
    from jnius import autoclass
    Environment = autoclass('android.os.Environment')
    ext_storage = Environment.getExternalStorageDirectory().getAbsolutePath()
    STATE_DIR = Path(ext_storage) / "KhanjarData"
except:
    # Fallback to current directory
    STATE_DIR = Path("khanjar_state")

STATE_DIR.mkdir(exist_ok=True, parents=True)
DAILY_LOG = STATE_DIR / "daily_trades.json"
ACCOUNT_FILE = STATE_DIR / "account.json"
PERF_FILE = STATE_DIR / "performance.json"
SIGNAL_FILE = STATE_DIR / "last_signals.json"

BLACKLIST = ['USDC', 'BUSD', 'FDUSD', 'TUSD', 'USDP', 'EUR', 'TRY', 'GBP', 'DAI']

# ========== کلاس‌های اصلی (مطابق اخرین۷.py) ==========

class USDTDominanceAnalyzer:
    def __init__(self, session):
        self.s = session

    def get_klines(self, symbol="BTCUSDT", tf="1h", limit=200):
        try:
            r = self.s.get(f"{BASE}/api/v3/klines?symbol={symbol}&interval={tf}&limit={limit}", timeout=10)
            return np.array(r.json(), dtype=float)
        except:
            return None

    def analyze(self):
        try:
            htf = self.get_klines(tf="1h", limit=200)
            ltf = self.get_klines(tf="15m", limit=200)

            if htf is None or ltf is None:
                return self._default()

            h, l, c = htf[:, 2], htf[:, 3], htf[:, 4]
            price = float(c[-1])

            sma20 = float(np.mean(c[-20:]))
            sma50 = float(np.mean(c[-50:]))
            trend = "صعودی" if sma20 > sma50 else "نزولی"

            atr_now = float(np.mean(np.abs(h[-14:] - l[-14:])))
            atr_avg = float(np.mean(np.abs(h[-50:] - l[-50:])))
            compression = atr_now < atr_avg * 0.85

            if trend == "صعودی" and compression:
                risk_mode = "RISK_ON"
                quality = 85
            elif trend == "صعودی":
                risk_mode = "RISK_ON"
                quality = 75
            elif trend == "نزولی":
                risk_mode = "RISK_OFF"
                quality = 40
            else:
                risk_mode = "NEUTRAL"
                quality = 50

            return {
                "price": price,
                "trend": trend,
                "compression_1h": compression,
                "risk_mode": risk_mode,
                "quality_score": quality,
                "valid": True
            }
        except:
            return self._default()

    def _default(self):
        return {
            "price": 0,
            "trend": "نامشخص",
            "compression_1h": False,
            "risk_mode": "NEUTRAL",
            "quality_score": 50,
            "valid": False
        }

class SessionFilter:
    @staticmethod
    def get_current_session():
        now = datetime.now(timezone.utc)
        hour = now.hour

        if 13 <= hour < 16:
            return "OVERLAP", "همپوشانی اروپا-آمریکا", 95, True
        elif 7 <= hour < 16:
            return "EUROPEAN", "اروپایی", 85, True
        elif 13 <= hour < 22:
            return "US", "آمریکایی", 90, True
        elif 0 <= hour < 9:
            return "ASIAN", "آسیایی", 60, False
        else:
            return "OFF", "خارج از ساعات", 40, False

class StateManager:
    def __init__(self):
        self.load()

    def load(self):
        today = datetime.now().strftime("%Y-%m-%d")

        if ACCOUNT_FILE.exists():
            with open(ACCOUNT_FILE) as f:
                self.account = json.load(f)
        else:
            self.account = {'balance': ACCOUNT_BALANCE_USD, 'initial': ACCOUNT_BALANCE_USD, 'date': today}
            self.save_account()

        if DAILY_LOG.exists():
            with open(DAILY_LOG) as f:
                data = json.load(f)
                if data.get('date') == today:
                    self.trades = data.get('trades', [])
                    self.pnl = data.get('pnl', 0.0)
                else:
                    self.trades = []
                    self.pnl = 0.0
        else:
            self.trades = []
            self.pnl = 0.0

        if PERF_FILE.exists():
            with open(PERF_FILE) as f:
                self.perf = json.load(f)
        else:
            self.perf = {'total': 0, 'wins': 0, 'losses': 0, 'consec_loss': 0, 'total_pnl': 0.0}

    def save_account(self):
        with open(ACCOUNT_FILE, 'w') as f:
            json.dump(self.account, f, indent=2)

    def save(self):
        today = datetime.now().strftime("%Y-%m-%d")
        with open(DAILY_LOG, 'w') as f:
            json.dump({'date': today, 'trades': self.trades, 'pnl': self.pnl}, f, indent=2)
        with open(PERF_FILE, 'w') as f:
            json.dump(self.perf, f, indent=2)
        self.save_account()

    def can_trade(self):
        if len(self.trades) >= MAX_TRADES_PER_DAY:
            return False, f"❌ محدودیت: {len(self.trades)}/{MAX_TRADES_PER_DAY}"
        if self.pnl <= MAX_DAILY_DRAWDOWN_PCT:
            return False, f"🚨 ضرر روزانه: {self.pnl:.2f}%"
        if self.perf['consec_loss'] >= MAX_CONSECUTIVE_LOSSES:
            return False, f"⚠️ کیل‌سوییچ: {MAX_CONSECUTIVE_LOSSES} ضرر"
        return True, "✅ آماده"

class LeverageOptimizer:
    @staticmethod
    def calc(atr_pct, conf, vol, rr):
        base = 3
        max_allowed = 5

        if vol == "SHOCK":
            max_vol = 2
        elif vol == "SILENT":
            max_vol = 5
        else:
            max_vol = 4

        conf_mult = 0.6 if conf < 1 else (0.8 if conf < 2 else 1.0)
        rr_mult = 0.8 if rr < 2 else 1.0

        opt = base * conf_mult * rr_mult
        opt = min(opt, max_vol, max_allowed)
        opt = max(opt, 1)

        risk = "ULTRA_SAFE" if opt <= 2 else ("SAFE" if opt <= 3 else "MODERATE")

        return {
            'lev': round(opt, 1),
            'risk': risk,
            'max': max_allowed,
            'reason': f"Vol:{vol} Conf:{conf:.1f} RR:{rr:.1f}"
        }

class PositionSizer:
    @staticmethod
    def calc(balance, risk_pct, entry, sl, lev):
        risk_amt = balance * (risk_pct / 100)
        sl_dist = abs(entry - sl) / entry

        if sl_dist == 0:
            return None

        pos = risk_amt / sl_dist
        exposure = pos * lev
        margin = pos

        max_pos = balance * (MAX_POSITION_PCT / 100) * lev
        if exposure > max_pos:
            exposure = max_pos
            margin = exposure / lev

        if margin < MIN_POSITION_USD:
            return None

        qty = exposure / entry

        return {
            'margin': round(margin, 2),
            'exposure': round(exposure, 2),
            'qty': round(qty, 8),
            'risk': round(risk_amt, 2),
            'risk_pct': risk_pct,
            'lev': lev,
            'sl_pct': round(sl_dist * 100, 2)
        }

    @staticmethod
    def split_tp(qty):
        return {
            'tp1': round(qty * 0.30, 8),
            'tp2': round(qty * 0.40, 8),
            'tp3': round(qty * 0.30, 8)
        }

class KhanjarSupremeV5:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.state = StateManager()
        self.sizer = PositionSizer()
        self.lev_opt = LeverageOptimizer()
        self.dominance = USDTDominanceAnalyzer(self.s)

    def pairs(self):
        try:
            r = self.s.get(f"{BASE}/api/v3/exchangeInfo", timeout=10)
            return [x['symbol'] for x in r.json()['symbols']
                    if x['status']=='TRADING' and x['quoteAsset']=='USDT'
                    and x['baseAsset'] not in BLACKLIST]
        except:
            return []

    def klines(self, sym, tf, lim=200):
        try:
            r = self.s.get(f"{BASE}/api/v3/klines?symbol={sym}&interval={tf}&limit={lim}", timeout=6)
            return np.array(r.json(), dtype=float)
        except:
            return None

    def stats24h(self, sym):
        try:
            r = self.s.get(f"{BASE}/api/v3/ticker/24hr?symbol={sym}", timeout=5)
            d = r.json()
            return {
                'vol': float(d['quoteVolume']),
                'price': float(d['lastPrice']),
                'bid': float(d['bidPrice']),
                'ask': float(d['askPrice'])
            }
        except:
            return None

    def liq_check(self, sym):
        st = self.stats24h(sym)
        if not st:
            return False, "No data"
        if st['vol'] < MIN_VOLUME_24H_USD:
            return False, "Low vol"
        spread = ((st['ask'] - st['bid']) / st['price']) * 100
        if spread > MAX_SPREAD_PCT:
            return False, "Wide spread"
        return True, f"Vol: ${st['vol']:,.0f}"

    def ema(self, data, period):
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        k = 2.0 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * k + ema[i-1]
        return ema

    def supertrend(self, h, l, c, length=10, factor=1.5):
        prev = np.roll(c, 1)
        prev[0] = c[0]
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))

        atr = np.zeros_like(c, dtype=float)
        atr[0] = tr[0]
        for i in range(1, len(c)):
            atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length

        hl2 = (h + l) / 2
        ub = hl2 + factor * atr
        lb = hl2 - factor * atr

        st = np.zeros_like(c, dtype=float)
        dir_arr = np.ones_like(c, dtype=float)

        st[0] = lb[0]
        for i in range(1, len(c)):
            if c[i] > st[i-1]:
                st[i] = max(lb[i], st[i-1])
                dir_arr[i] = 1
            else:
                st[i] = min(ub[i], st[i-1])
                dir_arr[i] = -1

        return st, dir_arr

    def signal(self, h, l, c, v):
        if len(c) < 210:
            return None

        ema7 = self.ema(c, 7)
        ema13 = self.ema(c, 13)
        ema200 = self.ema(c, 200)

        st_line, st_dir = self.supertrend(h, l, c)

        price = c[-1]
        swing_high = np.max(h[-30:])
        swing_low = np.min(l[-30:])

        # LONG
        if price > ema200[-1] and st_dir[-1] == 1:
            for i in range(1, 4):
                if ema7[-i-1] <= ema13[-i-1] and ema7[-i] > ema13[-i]:
                    return {
                        "direction": "LONG",
                        "st_line": st_line[-1],
                        "swing_high": swing_high,
                        "swing_low": swing_low,
                        "base_score": 70
                    }

        # SHORT
        if price < ema200[-1] and st_dir[-1] == -1:
            for i in range(1, 4):
                if ema7[-i-1] >= ema13[-i-1] and ema7[-i] < ema13[-i]:
                    return {
                        "direction": "SHORT",
                        "st_line": st_line[-1],
                        "swing_high": swing_high,
                        "swing_low": swing_low,
                        "base_score": 70
                    }

        return None

    def execution(self, h, l, c, sig):
        entry = c[-1]
        direction = sig["direction"]

        prev = np.roll(c[-20:], 1)
        prev[0] = c[-21] if len(c) > 21 else c[0]

        tr = np.maximum(h[-20:] - l[-20:], np.maximum(np.abs(h[-20:] - prev), np.abs(l[-20:] - prev)))
        atr = np.mean(tr[-14:])

        if atr == 0:
            return None

        atr_avg = np.mean(tr)
        if atr > atr_avg * 1.5:
            vol = "SHOCK"
        elif atr < atr_avg * 0.7:
            vol = "SILENT"
        else:
            vol = "NORMAL"

        sl_buffer = atr * 0.35

        if direction == "LONG":
            sl = min(sig["st_line"], sig["swing_low"]) - sl_buffer
            risk = entry - sl
            tp1 = entry + risk * 1.5
            tp2 = entry + risk * 2.2
            tp3 = entry + risk * 3.0
        else:
            sl = max(sig["st_line"], sig["swing_high"]) + sl_buffer
            risk = sl - entry
            tp1 = entry - risk * 1.5
            tp2 = entry - risk * 2.2
            tp3 = entry - risk * 3.0

        avg_tp = tp1 * 0.3 + tp2 * 0.4 + tp3 * 0.3
        rr = abs(avg_tp - entry) / risk

        if rr < 1.6:
            return None

        atr_pct = (atr / entry) * 100

        return {
            "entry": round(entry, 6),
            "sl": round(sl, 6),
            "tp1": round(tp1, 6),
            "tp2": round(tp2, 6),
            "tp3": round(tp3, 6),
            "rr": round(rr, 2),
            "vol": vol,
            "atr_pct": atr_pct
        }

    def analyze(self, sym, ctx):
        try:
            liq_ok, liq_msg = self.liq_check(sym)
            if not liq_ok:
                return None

            m1h = self.klines(sym, '1h', 200)
            m15 = self.klines(sym, '15m', 200)

            if m1h is None or m15 is None or len(m1h) < 200 or len(m15) < 100:
                return None

            h, l, c = m15[:, 2], m15[:, 3], m15[:, 4]
            v = m15[:, 5]

            sig = self.signal(h, l, c, v)
            if not sig:
                return None

            if ctx == "RISK_OFF" and sig["direction"] == "LONG":
                return None
            if ctx == "RISK_ON" and sig["direction"] == "SHORT":
                return None

            ex = self.execution(h, l, c, sig)
            if not ex:
                return None

            sma10 = np.mean(c[-10:])
            sma50 = np.mean(c[-50:])
            price = c[-1]
            raw_conf = abs(sma10 - sma50) / price * 1000
            conf = 0 if ex['rr'] < 2.0 or ex['vol'] != "NORMAL" else min(raw_conf, ex['rr'])

            lev_info = self.lev_opt.calc(ex['atr_pct'], conf, ex['vol'], ex['rr'])

            balance = self.state.account['balance']
            pos_info = self.sizer.calc(balance, RISK_PER_TRADE_PCT, ex['entry'], ex['sl'], lev_info['lev'])

            if not pos_info:
                return None

            tp_split = self.sizer.split_tp(pos_info['qty'])

            score = ex['rr'] * (1 + conf / 10)

            return {
                'sym': sym,
                'dir': sig['direction'],
                'entry': ex['entry'],
                'sl': ex['sl'],
                'tp1': ex['tp1'],
                'tp2': ex['tp2'],
                'tp3': ex['tp3'],
                'rr': ex['rr'],
                'conf': round(conf, 2),
                'vol': ex['vol'],
                'score': round(score, 2),
                'liq': liq_msg,
                'lev': lev_info,
                'pos': pos_info,
                'tp_split': tp_split
            }
        except:
            return None

    def scan(self):
        """Main scan function - returns results dict"""
        try:
            # Check if can trade
            ok, msg = self.state.can_trade()
            
            # Session
            session_code, session_name, session_quality, session_ok = SessionFilter.get_current_session()
            
            # Dominance
            dom = self.dominance.analyze()
            
            # Get pairs
            pairs = self.pairs()
            
            results = []
            
            if session_ok and ok:
                with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
                    for r in ex.map(lambda p: self.analyze(p, dom['risk_mode']), pairs):
                        if r:
                            results.append(r)
            
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Save to file
            output = {
                'timestamp': datetime.now().isoformat(),
                'market': dom,
                'session': {
                    'code': session_code,
                    'name': session_name,
                    'quality': session_quality,
                    'active': session_ok
                },
                'can_trade': ok,
                'trade_status': msg,
                'signals': results[:5]
            }
            
            with open(SIGNAL_FILE, 'w') as f:
                json.dump(output, f, indent=2)
            
            return output
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'market': None,
                'session': None,
                'signals': []
            }

# ========== Main Function for CLI/Service ==========

def run_scan():
    """Wrapper function for service/CLI"""
    engine = KhanjarSupremeV5()
    return engine.scan()

if __name__ == "__main__":
    result = run_scan()
    print(json.dumps(result, indent=2))
