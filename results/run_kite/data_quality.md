## Data quality

- sessions: 248 (2025-11-12 -> 2026-11-12)
- source rows read: 920,038; used: 892,799
- duplicate rows dropped: 0
- invalid price rows dropped: 0
- rows on non-session dates dropped: 550
- rows outside 09:15-15:15 dropped: 26,689
- continuous trading ends at 15:15 from 2026-08-03 (normally 15:30); every session uses the common 09:15-15:15 window so all days are comparable
- known special sessions ignored: 2026-11-08 (Muhurat trading (Sunday))

| symbol | full_sessions | partial_sessions | missing_sessions | bars_missing | pct_bars_missing |
|---|---|---|---|---|---|
| BHARTIARTL | 248 | 0 | 0 | 0 | 0.00 |
| HDFCBANK | 248 | 0 | 0 | 0 | 0.00 |
| ICICIBANK | 248 | 0 | 0 | 0 | 0.00 |
| INFY | 248 | 0 | 0 | 0 | 0.00 |
| ITC | 248 | 0 | 0 | 0 | 0.00 |
| KOTAKBANK | 248 | 0 | 0 | 0 | 0.00 |
| LT | 248 | 0 | 0 | 0 | 0.00 |
| RELIANCE | 248 | 0 | 0 | 0 | 0.00 |
| SBIN | 248 | 0 | 0 | 0 | 0.00 |
| TCS | 248 | 0 | 0 | 0 | 0.00 |
