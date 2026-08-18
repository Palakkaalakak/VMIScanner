@echo off
REM ============================================================
REM  VMI Scanner dashboard — LOCAL launcher (Windows)
REM  Double-click to run the dashboard on YOUR PC, where the
REM  AI Moat Evaluator tab can reach LM Studio at localhost:1234
REM  directly (no tunnel needed).
REM ============================================================
cd /d "%~dp0"
echo [1/2] Checking dependencies...
python -m pip install -q -r requirements.txt --upgrade-strategy only-if-needed
echo [2/2] Starting dashboard at http://localhost:8501 (Ctrl+C to stop)
python -m streamlit run streamlit_app.py --server.port 8501
pause
