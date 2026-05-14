@echo off
echo ============================================
echo    AI Image Studio - Starting Server
echo ============================================
echo.

:: Check Python
set PYTHON_EXE=
if exist "C:\Users\USER\AppData\Local\Python\bin\python.exe" (
    set PYTHON_EXE="C:\Users\USER\AppData\Local\Python\bin\python.exe"
) else if exist "C:\Users\USER\AppData\Local\Microsoft\WindowsApps\python.exe" (
    set PYTHON_EXE="C:\Users\USER\AppData\Local\Microsoft\WindowsApps\python.exe"
) else (
    where python >nul 2>&1 && set PYTHON_EXE=python
    if not defined PYTHON_EXE where python3 >nul 2>&1 && set PYTHON_EXE=python3
)

if not defined PYTHON_EXE (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo [INFO] Using Python: %PYTHON_EXE%

:: Check venv
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    %PYTHON_EXE% -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install dependencies
echo [INFO] Installing/checking dependencies...
python -m pip install -r requirements.txt

:: Create directories
if not exist "models" mkdir models
if not exist "outputs" mkdir outputs

echo.
echo [OK] Virtual environment ready
echo [OK] Dependencies installed
echo.
echo --------------------------------------------
echo  Server starting at: http://localhost:8000
echo  Web UI:             http://localhost:8000
echo  API Docs:           http://localhost:8000/docs
echo --------------------------------------------
echo.
echo Place your model in the "models" folder then restart!
echo.

:: Start server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

pause
