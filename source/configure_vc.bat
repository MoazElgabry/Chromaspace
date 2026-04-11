@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\build-ninja"
set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA_EXE=C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
set "MESON_PYTHON="
set "CUDA_NVCC="
set "CUDA_ROOT="

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul 2>nul
if errorlevel 1 (
  call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
)

for %%I in ("%ROOT%\.venv-textdeps\Scripts\python.exe" "%ROOT%\.meson-venv\Scripts\python.exe") do (
  if not defined MESON_PYTHON if exist "%%~fI" (
    "%%~fI" -c "import mesonbuild" >nul 2>nul
    if not errorlevel 1 set "MESON_PYTHON=%%~fI"
  )
)
if not defined MESON_PYTHON (
  python -c "import mesonbuild" >nul 2>nul
  if not errorlevel 1 (
    for /f "usebackq delims=" %%I in (`python -c "import sys; print(sys.executable)"`) do set "MESON_PYTHON=%%I"
  )
)
if not defined MESON_PYTHON (
  echo Missing Meson Python package for Chromaspace bundled text dependencies.
  echo Looked for a working interpreter in:
  echo   %ROOT%\.venv-textdeps\Scripts\python.exe
  echo   %ROOT%\.meson-venv\Scripts\python.exe
  echo   python on PATH
  echo Install it with: python -m pip install --user meson
  exit /b 1
)

for %%D in ("%CUDA_PATH_V12_9%" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" "%CUDA_PATH_V12_8%" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" "%CUDA_PATH_V11_8%" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" "%CUDA_PATH%" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1") do (
  if not defined CUDA_NVCC if exist "%%~fD\bin\nvcc.exe" (
    set "CUDA_ROOT=%%~fD"
    set "CUDA_NVCC=%%~fD\bin\nvcc.exe"
  )
)
if not defined CUDA_NVCC (
  echo Could not locate a supported CUDA toolkit.
  exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"

"%CMAKE_EXE%" -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="%MESON_PYTHON%" -DCUDAToolkit_ROOT="%CUDA_ROOT%" -DCMAKE_CUDA_COMPILER="%CUDA_NVCC%" -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler"
