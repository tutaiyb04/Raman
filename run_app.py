import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    # Hàm xử lý đường dẫn khi đóng gói vào file .exe
    basedir = getattr(sys, '_MEIPASS', os.getcwd())
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # Kích hoạt Streamlit để chạy giao diện chính của bạn
    # Thay 'app.py' bằng tên file giao diện bạn đang sử dụng
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())