@echo off
chcp 65001 >nul
echo ========================================
echo   OtakuWifu Copilot 快速部署脚本
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)

:: 创建虚拟环境
if not exist "venv" (
    echo [1/3] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
) else (
    echo [1/3] 虚拟环境已存在，跳过创建
)

:: 激活虚拟环境
echo [2/3] 激活虚拟环境...
call venv\Scripts\activate.bat

:: 安装依赖
echo [3/3] 安装核心依赖...
echo.
pip install fastapi uvicorn pywebview pyaudio webrtcvad requests mss Pillow numpy faster-whisper imageio-ffmpeg
echo.

if errorlevel 1 (
    echo [错误] 安装依赖失败，请检查网络连接
    pause
    exit /b 1
)

echo ========================================
echo   部署完成！
echo ========================================
echo.
echo 运行方式：
echo   1. 双击 run.bat 启动程序
echo   2. 或手动执行: venv\Scripts\activate ^&^& python web_app.py
echo.
pause
