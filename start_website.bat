@echo off
REM Starts the SAR LULC website on this laptop. Double-click, or run from cmd.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run the setup commands first ^(see README^).
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
set GEE_PROJECT=project-5ed4e2e0-4d5f-4a56-957
set PYTHONWARNINGS=ignore
streamlit run app.py
