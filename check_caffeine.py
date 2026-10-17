from app import create_app

app = create_app()
with app.app_context():
    from app.habits import HABITS
    from app.models import ProductivityLog

    print("Habit definition(s):")
    for h in HABITS:
        text = (str(h.get("key", "")) + str(h.get("field", "")) + str(h.get("name", ""))).lower()
        if "caff" in text:
            print("  ", {k: h.get(k) for k in ("key", "field", "name", "unit", "min", "max")})

    rows = ProductivityLog.query.filter(ProductivityLog.caffeine_mg.isnot(None)).all()
    values = [r.caffeine_mg for r in rows]
    if values:
        print(f"Stored caffeine values: {len(values)} rows, min {min(values)}, max {max(values)}")
        print("Values:", sorted(set(values)))
    else:
        print("No caffeine values stored yet.")