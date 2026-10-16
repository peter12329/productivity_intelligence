from flask import Blueprint

bp = Blueprint("insights", __name__, url_prefix="/insights")

from app.insights import routes  # noqa: E402,F401  (import last to avoid a circular import)