@echo off
rem LabelMyEye 眼底标注软件启动脚本
cd /d "%~dp0"
python run.py
if errorlevel 1 pause
