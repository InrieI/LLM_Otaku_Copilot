@echo off
setlocal
cd /d "%~dp0"
echo 正在启动 OtakuWifu 控制中心...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [错误] 找不到虚拟环境，请先初始化环境！
    pause
    exit /b
)
python web_app.py
pause
