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

    study_hours = db.Column(db.Float, nullable=False)
    sleep_hours = db.Column(db.Float, nullable=False)
    screen_time = db.Column(db.Float, nullable=False)
    exercise_minutes = db.Column(db.Integer, nullable=False)
    caffeine_mg = db.Column(db.Integer, nullable=False)
    stress_level = db.Column(db.Integer, nullable=False)
    break_count = db.Column(db.Integer, nullable=False)
    study_sessions = db.Column(db.Integer, nullable=False)
    self_rating = db.Column(db.Integer, nullable=True)   # optional 0-100, the ML target later
    notes = db.Column(db.Text, nullable=True)

    user = db.relationship("User", backref="logs")

    def goals_met(self):
        return sum([
            self.study_hours >= 3,
            self.sleep_hours >= 7,
            self.exercise_minutes >= 20,
            self.screen_time < 6,
            self.break_count >= 1,
        ])

    def level(self):
        """0-4 colour level for the heatmap."""
        g = self.goals_met()
        return 4 if g == 5 else 3 if g == 4 else 2 if g >= 2 else 1



@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))