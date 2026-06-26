@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\build-ninja"
set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul 2>nul
if errorlevel 1 (
  call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
)

call "%~dp0configure_vc.bat" || exit /b 1
"%CMAKE_EXE%" --build "%BUILD_DIR%" --target Chromaspace_Bundle || exit /b 1

set "BUNDLE_ROOT=%ROOT%\bundle\Chromaspace.ofx.bundle"
set "BUNDLE_WIN64=%BUNDLE_ROOT%\Contents\Win64"
set "BUNDLE_RESOURCES=%BUNDLE_ROOT%\Contents\Resources"

if not exist "%BUNDLE_WIN64%\Chromaspace.ofx" (
  echo Bundle validation failed: missing Chromaspace.ofx
  exit /b 1
)
if not exist "%BUNDLE_WIN64%\Chromaspace_CubeViewer.exe" (
  echo Bundle validation failed: missing Chromaspace_CubeViewer.exe
  exit /b 1
)
if not exist "%BUNDLE_WIN64%\OpenSans-Regular.ttf" (
  echo Bundle validation failed: missing OpenSans-Regular.ttf
  exit /b 1
)
if not exist "%BUNDLE_RESOURCES%\chromaspace_resolve_bridge.py" (
  echo Bundle validation failed: missing chromaspace_resolve_bridge.py
  exit /b 1
)
if not exist "%BUNDLE_RESOURCES%\com.moazelgabry.chromaspace.png" (
  echo Bundle validation failed: missing plugin icon
  exit /b 1
)

echo Install-ready bundle created at:
echo   %BUNDLE_ROOT%
