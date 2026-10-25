from datetime import date, datetime, timezone

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app.extensions import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    theme = db.Column(db.String(30), nullable=False, default="midnight-purple")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class ProductivityLog(db.Model):
    __tablename__ = "productivity_logs"
    __table_args__ = (db.UniqueConstraint("user_id", "log_date", name="uq_user_date"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    log_date = db.Column(db.Date, nullable=False, default=date.today)

    # Original habits (study_hours stores deep work hours, caffeine_mg stores a 0-10 level)
    study_hours = db.Column(db.Float, nullable=True)
    sleep_hours = db.Column(db.Float, nullable=True)
    caffeine_mg = db.Column(db.Integer, nullable=True)
    stress_level = db.Column(db.Integer, nullable=True)
    study_sessions = db.Column(db.Integer, nullable=True)
    journaled = db.Column(db.Boolean, nullable=True)

    # New goal habits
    disruptions = db.Column(db.Integer, nullable=True)       # meetings / interruptions
    junk_food = db.Column(db.Integer, nullable=True)         # 0 = no, 1 = yes

    # Context inputs (no goal)
    task_difficulty = db.Column(db.Integer, nullable=True)   # 1-5
    free_hours = db.Column(db.Float, nullable=True)
    day_type = db.Column(db.String(16), nullable=True)       # class / free / deadline
    weather = db.Column(db.String(16), nullable=True)        # sunny / cloudy / rainy

    user = db.relationship("User", backref="logs")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))