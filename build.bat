@echo off
REM ---------------------------------------------------------------------------
REM  GoblinScript build helper (user machine)
REM
REM  The Rust toolchain lives on H: (H:\RustEnv) because C: is nearly full, and
REM  so does the cargo target dir. This script puts the REAL toolchain bin ahead
REM  of the rustup shims: the shims under H:\RustEnv\cargo\bin need the rustup
REM  proxy to resolve RUSTUP_HOME in an environment this shell does not always
REM  give them, and they exit silently when it cannot.
REM
REM  Usage:
REM      build.bat            -> release build, DirectML (any DX12 GPU)
REM      build.bat debug      -> debug build
REM      build.bat cuda       -> release build with the CUDA provider
REM      build.bat check      -> cargo check only (fast, no codegen)
REM ---------------------------------------------------------------------------
setlocal

set "RUSTUP_HOME=H:\RustEnv\rustup"
set "CARGO_HOME=H:\RustEnv\cargo"
set "TC=H:\RustEnv\rustup\toolchains\stable-x86_64-pc-windows-msvc\bin"
set "PATH=%TC%;H:\RustEnv\cargo\bin;%PATH%"
set "CARGO_TARGET_DIR=H:\RustEnv\target"
REM The crates come through the Tsinghua mirror and the Clash proxy (see
REM %CARGO_HOME%\config.toml); the dist server is not needed for a build.
set "RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup"
REM rc.exe, so build.rs can embed the app icon and manifest. It is not on PATH
REM by default, and without it the build still succeeds -- the exe just gets
REM the default icon and "goblin resources not embedded" as a warning.
if exist "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64" (
    set "PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;%PATH%"
)

cd /d "%~dp0"

if /I "%~1"=="check" (
    echo [build] cargo check
    "%TC%\cargo.exe" check --features embed
    goto :done
)
if /I "%~1"=="debug" (
    echo [build] debug ^(DirectML^)
    "%TC%\cargo.exe" build
    goto :done
)
if /I "%~1"=="cuda" (
    echo [build] release ^(CUDA^)
    "%TC%\cargo.exe" build --release --features embed,cuda
    goto :done
)

echo [build] release ^(DirectML, embedded bundle^)
"%TC%\cargo.exe" build --release --features embed

:done
if errorlevel 1 (
    echo.
    echo [build] FAILED
    exit /b 1
)
echo.
echo [build] done -^> %CARGO_TARGET_DIR%\release\goblinscript.exe
endlocal
