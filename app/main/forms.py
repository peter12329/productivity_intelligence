from datetime import date

from flask_wtf import FlaskForm
from wtforms import DateField, FloatField, SubmitField
from wtforms.validators import InputRequired, ValidationError


def _check_date(field):
    if field.data > date.today():
        raise ValidationError("Date cannot be in the future.")
    if field.data.year < 2020:
        raise ValidationError("Date is too far in the past.")


class HabitForm(FlaskForm):
    """One habit value for one day. The range check happens in the route."""
    log_date = DateField("Date", default=date.today, validators=[InputRequired()])
    value = FloatField("Value", validators=[InputRequired()])
    submit = SubmitField("Save")

    def validate_log_date(self, field):
        _check_date(field)