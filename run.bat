@echo off
chcp 65001 >nul

if not exist "venv\Scripts\activate.bat" (
    echo [错误] 未找到虚拟环境，请先运行 deploy.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python web_app.py
