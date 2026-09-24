@echo off
cd /d "%~dp0"
echo Starting TensorBoard for the Reinforcement Learning Agent...
echo The dashboard will be available at http://localhost:6006
echo.
call .\env\Scripts\activate.bat
tensorboard --logdir=tensorboard_logs/ --port=6006
pause
