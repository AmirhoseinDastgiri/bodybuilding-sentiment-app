@echo off
REM Starts the Iron Coach web server and opens it in the browser.
REM Keep this window open while you chat; close it (or press Ctrl+C) to stop.
cd /d "%~dp0"
if not exist "bodyvenv\Scripts\python.exe" (
    echo bodyvenv not found. Create it with:  python -m venv bodyvenv
    echo then:  bodyvenv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo Starting Iron Coach... the browser will open when the server is ready.
bodyvenv\Scripts\python.exe server.py --open
pause
