@echo off

IF EXIST "venv" (
    call "./venv/Scripts/activate.bat"
) ELSE (
    python -m venv venv
    call "./venv/Scripts/activate.bat"
    pip install -r requirements.txt
)

python ./app.py
call "./venv/Scripts/deactivate.bat"
