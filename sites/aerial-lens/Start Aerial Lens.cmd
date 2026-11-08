@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:8766 after the server starts.
node serve.mjs
if errorlevel 1 pause
