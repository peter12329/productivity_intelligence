"""Habit definitions. To add a habit, copy the Study entry and change it."""

HABITS = [
    {
        "key": "study",
        "name": "Study 3h+",
        "label": "Study hours",
        "field": "study_hours",    
        "unit": "hours",
        "min": 0, "max": 24, "step": 0.5,
        "cast": float,
        "color": "#6f42c1",
        "met": lambda v: v >= 3,     
    },
    {
        "key": "caffeine",
        "name": "Caffeine under 300mg+",
        "label": "Caffeine",
        "field": "caffeine_mg",       
        "unit": "out of 10",
        "min": 0, "max": 10, "step": 1,
        "cast": int,
        "color": "#6c757d",
        "met": lambda v: v >= 5,      
    },
    {
        "key": "stress", "name": "Stress 5 or lower", "label": "Stress level",
        "field": "stress_level", "unit": "out of 10",
        "min": 1, "max": 10, "step": 1, "cast": int,
        "color": "#6c757d", "met": lambda v: v <= 5,
    },
    {
        "key": "sessions", "name": "2+ study sessions", "label": "Study sessions",
        "field": "study_sessions", "unit": "sessions",
        "min": 0, "max": 50, "step": 1, "cast": int,
        "color": "#0dcaf0", "met": lambda v: v >= 2,
    },
        
        {
        "key": "journal",
        "name": "Journaling",
        "type": "check",          
        "field": "journaled",
        "color": "#6610f2",
        "met": lambda v: v is True,
    },

]

BY_KEY = {h["key"]: h for h in HABITS}


def is_met(habit, log):
    """True = goal met, False = logged but not met, None = no data."""
    value = getattr(log, habit["field"])
    return None if value is None else bool(habit["met"](value))


def goals_met(log):
    return sum(1 for h in HABITS if is_met(h, log))


def level(log):
    """0 = no data, 1-4 = share of goals met (4 = full day)."""
    states = [is_met(h, log) for h in HABITS]
    if all(s is None for s in states):
        return 0
    frac = sum(1 for s in states if s) / len(HABITS)
    return 4 if frac == 1 else 3 if frac >= 0.7 else 2 if frac >= 0.4 else 1