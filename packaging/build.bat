@echo off
rem Builds the stand-alone NSSA, the NSF Surface and Spectroscopy Analyzer (Windows):
rem   packaging\dist\NSSA\NSSA.exe   (the folder can be copied and run anywhere)
rem   packaging\Output\NSSA-Setup-<version>.exe          (only if Inno Setup 6 is installed)
rem Needs Python 3.10 or newer on the PATH. The build environment (only the packages of requirements-build.txt)
rem and the temporary build files are kept outside this folder, in %LOCALAPPDATA%\NSSA\Build:
rem a synced folder (Dropbox, OneDrive) locks files while pip and PyInstaller write them, which breaks the build.
setlocal
cd /d "%~dp0"
set "BUILDROOT=%LOCALAPPDATA%\NSSA\Build"
set "VENV=%BUILDROOT%\venv"
set "PY=%VENV%\Scripts\python.exe"

where python >nul 2>nul || (echo Python was not found on the PATH. & goto :error)
if not exist "%BUILDROOT%" mkdir "%BUILDROOT%"

rem a build environment whose pip is broken (e.g. an interrupted pip upgrade) is made again
if not exist "%PY%" goto :make_venv
"%PY%" -m pip --version >nul 2>nul && goto :have_venv
echo The build environment is damaged, making it again ...
rmdir /s /q "%VENV%"
:make_venv
echo Creating the build environment in %VENV% ...
python -m venv "%VENV%" || goto :error
"%PY%" -m pip --version >nul 2>nul || (echo pip is missing in the new environment. & goto :error)
:have_venv

rem NSFopen from the folder next to NSSA (Nanosurf\NSFopen) when it is there: the version being developed, which may
rem not be on PyPI yet. Without that folder, requirements-build.txt takes it from PyPI.
set "NSFOPEN_SRC=%~dp0..\..\NSFopen"
if exist "%NSFOPEN_SRC%\setup.py" echo Installing NSFopen from %NSFOPEN_SRC% ...
if exist "%NSFOPEN_SRC%\setup.py" "%PY%" -m pip install --disable-pip-version-check --force-reinstall --no-deps "%NSFOPEN_SRC%" || goto :error

echo Installing the packages ...
"%PY%" -m pip install --disable-pip-version-check -r requirements-build.txt || goto :error

echo Building ...
"%PY%" -m PyInstaller --noconfirm --clean --workpath "%BUILDROOT%\work" --distpath "%~dp0dist" NSSA.spec || goto :error

echo Checking the build ...
"dist\NSSA\NSSA-batch.exe" --selftest || goto :error

set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" goto :no_inno
"%PY%" -c "import sys; sys.path.insert(0, r'..\app_NSSA'); import appinfo; print(appinfo.VERSION)" > "%BUILDROOT%\version.txt" || goto :error
set /p VERSION=<"%BUILDROOT%\version.txt"
"%ISCC%" /DAppVersion=%VERSION% installer.iss || goto :error
echo Installer: packaging\Output\NSSA-Setup-%VERSION%.exe
goto :done

:no_inno
echo Inno Setup 6 not found: no installer made. The folder dist\NSSA works without it.

:done
echo.
echo Done: packaging\dist\NSSA\NSSA.exe
exit /b 0

:error
echo.
echo The build failed, see the messages above.
exit /b 1
