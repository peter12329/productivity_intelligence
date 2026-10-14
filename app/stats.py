import pandas as pd
from app.habits import HABITS, goals_met, is_met
from app.models import ProductivityLog


def load_df(user):
    logs = (ProductivityLog.query.filter_by(user_id=user.id)
            .order_by(ProductivityLog.log_date).all())
    rows = []
    for l in logs:
        row = {"date": pd.Timestamp(l.log_date),
               "score": goals_met(l) / len(HABITS) * 100}
        for h in HABITS:
            row[h["field"]] = getattr(l, h["field"])
            row[h["key"] + "_met"] = is_met(h, l)      # True / False / None
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


def habit_stats(df):
    out = []
    for h in HABITS:
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


def correlations(df):
    fields = [h["field"] for h in HABITS]
    num = df[fields].apply(pd.to_numeric, errors="coerce")
    c = num.corrwith(df["score"]).dropna()
    c = c.reindex(c.abs().sort_values(ascending=False).index)
    return {k: round(float(v), 3) for k, v in c.items()}


def by_weekday(df):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g = df.groupby(df["date"].dt.day_name())["score"].mean().reindex(order).dropna()
    return {k: round(float(v), 1) for k, v in g.items()}


def dashboard_stats(user):
    df = load_df(user)
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
        "habits": habit_stats(df),
        "trend": {
            "dates": [str(d.date()) for d in df["date"]],
            "score": [round(float(v), 1) for v in df["score"]],
            "rolling": [round(float(v), 1) for v in df["score"].rolling(7, min_periods=1).mean()],
        },
        "correlations": correlations(df),
        "weekday": by_weekday(df),
    }