"""Disable external telemetry in tests; telemetry tests can opt in explicitly."""

import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
