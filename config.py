import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///app.db")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # Admin login credentials (single operator). Set both in .env in production.
    ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "change-me-in-production")

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    KIE_API_KEY = os.environ.get("KIE_API_KEY", "")

    SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
    STORAGE_PATH = os.environ.get("STORAGE_PATH", "storage/designs")

    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── Connection pool ──────────────────────────────────────────────────────
    # Bound the pool so the web + celery processes can never exhaust Postgres'
    # max_connections (default 100). pool_pre_ping transparently discards a dead
    # connection (fixes "unexpected EOF on client connection") before handing it
    # out; pool_recycle proactively rotates connections older than 30 min so they
    # don't get reaped server-side. Budget: per process up to
    # pool_size + max_overflow = 15 connections.
    #   web (gunicorn -w 2)   → 2 × 15 = 30
    #   celery (--concurrency=1) →     15
    # Well under 100 even with headroom for the transient schema-init connection.
    if DATABASE_URL.startswith("postgres"):
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 1800,
            "pool_pre_ping": True,
        }
    else:
        # SQLite (local dev fallback) uses a non-queue pool — only pre_ping applies.
        SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}

def get_config():
    env = os.environ.get("FLASK_ENV", "default")
    return config_map.get(env, DevelopmentConfig)
