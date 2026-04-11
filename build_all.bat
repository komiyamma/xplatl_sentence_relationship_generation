@echo off
cd /d "%~dp0"
python build_index_cli.py || exit /b %errorlevel%
python score_related_all_cli.py || exit /b %errorlevel%
