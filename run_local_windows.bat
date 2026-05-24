@echo off
cd /d "%~dp0"

echo ============================================
echo  Dashboard Apex Sentiment - Local Windows
echo ============================================
echo.

echo Mengecek Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.11 belum terdeteksi.
    echo Install Python 3.11 64-bit terlebih dahulu dari python.org.
    echo Saat instalasi, centang Add python.exe to PATH.
    echo.
    echo Setelah itu jalankan ulang file ini.
    pause
    exit /b 1
)

if not exist .venv (
    echo Membuat virtual environment dengan Python 3.11...
    py -3.11 -m venv .venv
)

echo Mengaktifkan virtual environment...
call .venv\Scripts\activate.bat

echo Upgrade pip, setuptools, dan wheel...
python -m pip install --upgrade pip setuptools wheel

echo Install library dari requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Install library gagal.
    echo Pastikan Python yang digunakan adalah 3.11, bukan 3.14.
    echo Cek dengan perintah: python --version
    pause
    exit /b 1
)

echo.
echo Menjalankan dashboard...
streamlit run app.py
pause
