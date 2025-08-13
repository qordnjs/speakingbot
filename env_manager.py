import os
import sys
import subprocess
from pathlib import Path

# Thư mục lưu file backup packages offline
BACKUP_DIR = Path("packages_backup")
# File lưu danh sách gói + phiên bản
REQ_FILE = Path("requirements.txt")

def run_cmd(cmd):
    """Chạy lệnh shell và hiện output trực tiếp"""
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"Lệnh lỗi: {cmd}")

def backup_env():
    """Backup danh sách gói + file cài offline"""
    print("📦 Đang xuất danh sách gói vào requirements.txt...")
    run_cmd(f"pip freeze > {REQ_FILE}")

    print("💾 Đang tải gói để cài offline...")
    BACKUP_DIR.mkdir(exist_ok=True)
    run_cmd(f"pip download -r {REQ_FILE} -d {BACKUP_DIR}")

    print("✅ Backup hoàn tất!")
    print(f" - Danh sách gói: {REQ_FILE}")
    print(f" - File offline: {BACKUP_DIR}")

def restore_env():
    """Restore môi trường từ file backup"""
    if not REQ_FILE.exists():
        sys.exit("❌ Không tìm thấy requirements.txt để restore.")

    if BACKUP_DIR.exists():
        print("♻️ Đang cài gói từ file offline...")
        run_cmd(f"pip install --no-index --find-links={BACKUP_DIR} -r {REQ_FILE}")
    else:
        print("🌐 Không tìm thấy file offline, đang cài qua internet...")
        run_cmd(f"pip install -r {REQ_FILE}")

    print("✅ Restore hoàn tất!")

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("backup", "restore"):
        print("Cách dùng:")
        print("  python env_manager.py backup   # Lưu môi trường")
        print("  python env_manager.py restore  # Khôi phục môi trường")
        sys.exit(1)

    action = sys.argv[1]
    if action == "backup":
        backup_env()
    elif action == "restore":
        restore_env()

if __name__ == "__main__":
    main()
