import streamlit.web.cli as stcli
import os, sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resource_path("app.py"), # Thay bằng tên file giao diện của bạn
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())