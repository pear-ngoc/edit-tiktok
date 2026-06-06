@echo off
setlocal
if "%~1"=="" (
  py -3.11 main.py
) else (
  py -3.11 main.py %*
)
