@echo off
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%venv\Scripts\python.exe"

if exist "%VENV_PY%" (
  "%VENV_PY%" "%ROOT%main.py"
) else (
  python "%ROOT%main.py"
)
