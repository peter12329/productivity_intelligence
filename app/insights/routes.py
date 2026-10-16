from flask import Blueprint, flash, render_template, request
from flask_login import current_user, login_required

from app.habits import HABITS            # adjust to where habits.py really is
from app.models import ProductivityLog   # adjust to your model's real import
from ml.predict import (ModelsNotTrainedError, data_stage, model_info,
                        predict_day, predict_for_log)

insights = Blueprint("insights", __name__, url_prefix="/insights")

# ... _parse_form() unchanged ...

@insights.route("/", methods=["GET", "POST"])     # was @bp.route
@login_required
def index():
    ...  # unchanged

@insights.route("/log/<int:log_id>")              # was @bp.route
@login_required
def estimate_log(log_id):
    ...  # unchanged