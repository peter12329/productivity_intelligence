from datetime import date

from flask_wtf import FlaskForm
from wtforms import DateField, FloatField, IntegerField, TextAreaField, SubmitField
from wtforms.validators import InputRequired, NumberRange, Optional, Length


class LogForm(FlaskForm):
    log_date = DateField("Date", default=date.today, validators=[InputRequired()])
    study_hours = FloatField("Study hours", validators=[InputRequired(), NumberRange(0, 24)])
    sleep_hours = FloatField("Sleep hours", validators=[InputRequired(), NumberRange(0, 24)])
    screen_time = FloatField("Screen time (hours)", validators=[InputRequired(), NumberRange(0, 24)])
    exercise_minutes = IntegerField("Exercise (minutes)", validators=[InputRequired(), NumberRange(0, 600)])
    caffeine_mg = IntegerField("Caffeine (mg)", validators=[InputRequired(), NumberRange(0, 1000)])
    stress_level = IntegerField("Stress (1-10)", validators=[InputRequired(), NumberRange(1, 10)])
    break_count = IntegerField("Breaks", validators=[InputRequired(), NumberRange(0, 50)])
    study_sessions = IntegerField("Study sessions", validators=[InputRequired(), NumberRange(0, 50)])
    self_rating = IntegerField("Rate your day (0-100, optional)", validators=[Optional(), NumberRange(0, 100)])
    notes = TextAreaField("Notes (optional)", validators=[Optional(), Length(max=500)])
    submit = SubmitField("Save day")