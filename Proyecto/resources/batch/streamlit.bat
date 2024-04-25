@echo off
cd..
cd..
setlocal
set PROJECTPATH=%cd%
set PYTHONPATH=%PYTHON310_HOME%;%cd%
echo "Enviroment variable at '%PYTHONPATH%"
set MAINPATH=%PROJECTPATH%\streamlit_app.py
echo "Starting the app at '%MAINPATH%'"
%PYTHON310_HOME%\python -m streamlit run "%MAINPATH%"
endlocal