from flask import Flask
from flask_wtf import CSRFProtect

from config import Config
from app.extensions import db, login_manager


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    login_manager.init_app(app)
    CSRFProtect(app)

    from app import models  # noqa: F401

    from app.main.routes import main_bp
    app.register_blueprint(main_bp)

    from app.auth.routes import auth_bp
    app.register_blueprint(auth_bp)

    from app.insights.routes import insights
    app.register_blueprint(insights)

    with app.app_context():
        db.create_all()
        _ensure_columns()

    return app


# Columns added after the first release. create_all() never alters existing
# tables, so older databases get them here: (table, column, SQL type and default).
_ADDED_COLUMNS = [
    ("users", "theme", "VARCHAR(30) NOT NULL DEFAULT 'midnight-purple'"),
    ("productivity_logs", "exercise_minutes", "INTEGER"),
]


def _ensure_columns():
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    tables = set(inspector.get_table_names())
    for table, column, ddl in _ADDED_COLUMNS:
        if table not in tables:
            continue
        if column in {c["name"] for c in inspector.get_columns(table)}:
            continue
        with db.engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
