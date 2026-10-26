"""Run tests with auth off, whatever the developer's local .env says (env vars beat .env)."""
import os

os.environ["AUTH_MODE"] = "none"
