"""Streamlit Community Cloud entrypoint (share.streamlit.io).

Deploy settings on https://share.streamlit.io/deploy :
  Repository : Palakkaalakak/VMIScanner
  Branch     : main
  Main file  : streamlit_app.py

This just runs the real dashboard (scanner/webapp_ui.py) unchanged, so
local (`streamlit run scanner/webapp_ui.py` or run_dashboard.bat) and
hosted deployments share one codebase.

NOTE for the AI Moat Evaluator tab on the HOSTED site: the cloud server
cannot reach http://localhost:1234 on your PC. In LM Studio enable
Developer -> Settings -> "Serve on Local Network" and expose it with a
tunnel (e.g. `cloudflared tunnel --url http://localhost:1234` or ngrok),
then paste the public URL + /v1 into the tab's server-URL box.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# Make both `import scanner.x` and the intra-folder `import x` styles work.
for p in (ROOT, os.path.join(ROOT, "scanner")):
    if p not in sys.path:
        sys.path.insert(0, p)

runpy.run_path(os.path.join(ROOT, "scanner", "webapp_ui.py"),
               run_name="__main__")
