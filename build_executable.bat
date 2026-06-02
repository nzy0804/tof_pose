@echo off
REM PyInstaller build script for gRPC server
REM This generates a standalone executable that can be deployed on Linux/Windows servers

setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo Building executable...
set PYI_CMD=python -m PyInstaller
where conda >nul 2>nul
if not errorlevel 1 (
  REM conda run captures output by default; use --no-capture-output to stream PyInstaller logs.
  set PYI_CMD=conda run -n maixpose --no-capture-output python -m PyInstaller
)

echo Using: !PYI_CMD!
!PYI_CMD! ^
  --noconfirm ^
  --clean ^
  --noupx ^
  --onefile ^
  --console ^
  --name maixsense-grpc-server ^
  --distpath dist ^
  --workpath build ^
  --specpath . ^
  --paths src ^
  --collect-submodules tof_pose ^
  --hidden-import=tof_pose.realtime_service ^
  --hidden-import=tof_pose.paths ^
  --hidden-import=tof_pose.person_distance ^
  --hidden-import=tof_pose.pose_drawing ^
  --hidden-import=ultralytics ^
  --hidden-import=cv2 ^
  --hidden-import=grpc ^
  --hidden-import=numpy ^
  --add-data "assets;assets" ^
  scripts/grpc_server.py

if %ERRORLEVEL% EQU 0 (
  echo.
  echo Build successful! Executable: dist\maixsense-grpc-server.exe
  echo.
  echo To run locally:
  echo   .\dist\maixsense-grpc-server.exe --host 0.0.0.0 --port 50052
  echo.
  echo Notes:
  echo   - This script builds a Windows .exe only.
  echo   - To build a Linux executable, use WSL/Linux and run: bash build_executable_wsl.sh
  echo.
  echo Deploy (Linux server) overview:
  echo   1. Build Linux executable via build_executable_wsl.sh, then copy dist/maixsense-grpc-server to /opt/maixsense/
  echo   2. Create systemd service (see deploy/maixsense-grpc.service)
  echo   3. Start: sudo systemctl start maixsense-grpc
  echo.
) else (
  echo Build failed!
  pause
  exit /b 1
)
