"""主启动器：启动 Web 控制面板。请直接运行 web_app.py 或 run_webapp.bat。"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "web_app.py")])
