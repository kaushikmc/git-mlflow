import asyncio
import sys
from mlflow.cli import server

# This is the crucial part that fixes the Windows OSError
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# This line starts the MLflow server
server()