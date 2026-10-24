import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")          # values already set in the environment win


def _secret_key():
    """SECRET_KEY from the environment / .env, else one saved in instance/.

    A key that changes on every start logs everybody out, so when none is set a
    random one is created once and reused (instance/ is git-ignored).
    """
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    path = ROOT / "instance" / ".secret_key"
    try:
        return path.read_text().strip()
    except OSError:
        pass
    key = secrets.token_hex(32)
    try:
        path.parent.mkdir(exist_ok=True)
        path.write_text(key)
    except OSError:
        pass                         # read-only disk: fall back to a per-run key
    return key


class Config:
    SECRET_KEY = _secret_key()
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///ppia.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
