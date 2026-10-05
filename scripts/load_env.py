"""Load repo-root .env into os.environ (does not override existing vars)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> Path:
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return env_path
    try:
        from dotenv import load_dotenv
    except ImportError as e:
        raise SystemExit("python-dotenv is required: pip install python-dotenv") from e
    load_dotenv(env_path, override=False)
    return env_path


if __name__ == "__main__":
    p = load_env()
    import os
    keys = [
        "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN",
        "GITHUB_TOKEN", "HF_TOKEN", "CRYPTOCOMPARE_API_KEY",
        "NEWSAPI_KEY", "FOREX_NEWS_API_TOKEN", "PARSEBOT",
        "POLYGON_API_KEY", "FXMACRODATA_API_KEY",
    ]
    print(f"Loaded: {p if p.exists() else '(missing .env)'}")
    for k in keys:
        v = os.environ.get(k, "")
        print(f"  {k}: {'set' if v.strip() else 'empty'}")
