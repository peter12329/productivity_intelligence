import pandas as pd
from app.habits import HABITS, is_met
from app.models import ProductivityLog


def load_df(user, habits=None):
    """One row per day on which at least one of `habits` was logged.

    `habits` is the list of goal habits the user tracks (default: all of them).
    The score is the share of THOSE habits whose goal was met, the same rule the
    home heatmap uses. A day with only context logged (weather, day type, ...)
    has no tracked habit data, so it is skipped instead of counting as 0%.
    """
    habits = list(HABITS if habits is None else habits)
    if not habits:
        return pd.DataFrame()
    logs = (ProductivityLog.query.filter_by(user_id=user.id)
            .order_by(ProductivityLog.log_date).all())
    rows = []
    for l in logs:
        states = [is_met(h, l) for h in habits]
        if all(s is None for s in states):
            continue
        row = {"date": pd.Timestamp(l.log_date),
               "score": sum(1 for s in states if s) / len(habits) * 100}
        for h, s in zip(habits, states):
            row[h["field"]] = getattr(l, h["field"])
            row[h["key"] + "_met"] = s                  # True / False / None
        rows.append(row)
    return pd.DataFrame(rows)


def _streaks(met, dates):
    best = cur = 0
    prev = None
    for ok, d in zip(met, dates):
        if not (ok == True):          # False, None or NaN breaks the streak
            cur = 0
        else:
            cur = cur + 1 if (cur and (d - prev).days == 1) else 1
        best = max(best, cur)
        prev = d
    return cur, best


def habit_stats(df, habits=None):
    out = []
    for h in (HABITS if habits is None else habits):
        met = df[h["key"] + "_met"]
        known = met.dropna()
        cur, best = _streaks(met, df["date"])
        vals = pd.to_numeric(df[h["field"]], errors="coerce").dropna()
        out.append({
            "key": h["key"], "name": h["name"], "color": h["color"],
            "average": round(float(vals.mean()), 2) if len(vals) else None,
            "met_rate": round(float(known.astype(bool).mean()) * 100, 1) if len(known) else 0,
            "current_streak": cur, "best_streak": best,
        })
    return out


def correlations(df, habits=None):
    fields = [h["field"] for h in (HABITS if habits is None else habits)]
    num = df[fields].apply(pd.to_numeric, errors="coerce")
    c = num.corrwith(df["score"]).dropna()
    c = c.reindex(c.abs().sort_values(ascending=False).index)
    return {k: round(float(v), 3) for k, v in c.items()}


def by_weekday(df):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g = df.groupby(df["date"].dt.day_name())["score"].mean().reindex(order).dropna()
    return {k: round(float(v), 1) for k, v in g.items()}


def dashboard_stats(user, habits=None):
    """Dashboard numbers for `user`, computed over `habits` (the ones they track)."""
    habits = list(HABITS if habits is None else habits)
    df = load_df(user, habits)
    if df.empty:
        return {"empty": True}
    return {
        "empty": False,
        "summary": {
            "total_logs": len(df),
            "avg_score": round(float(df["score"].mean()), 1),
            "best_day": str(df.loc[df["score"].idxmax(), "date"].date()),
            "best_score": round(float(df["score"].max()), 1),
        },
        "habits": habit_stats(df, habits),
        "trend": {
            "dates": [str(d.date()) for d in df["date"]],
            "score": [round(float(v), 1) for v in df["score"]],
            "rolling": [round(float(v), 1) for v in df["score"].rolling(7, min_periods=1).mean()],
        },
        "correlations": correlations(df, habits),
        "weekday": by_weekday(df),
    }
