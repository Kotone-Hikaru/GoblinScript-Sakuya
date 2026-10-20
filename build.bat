@echo off
REM ---------------------------------------------------------------------------
REM  GoblinScript build helper (Windows)
REM
REM  Usage:
REM      build.bat            -> release build, DirectML (any DX12 GPU)
REM      build.bat debug      -> debug build
REM      build.bat cuda       -> release build with the CUDA provider
REM      build.bat check      -> cargo check only (fast, no codegen)
REM
REM  A plain `cargo build` works too; this only adds the two things a Windows
REM  build of this tree usually wants: rc.exe (so build.rs embeds the icon and
REM  the manifest) and nothing else. Without rc.exe the build still succeeds --
REM  the exe just gets the default icon and a "goblin resources not embedded"
REM  warning.
REM
REM  Override the toolchain with CARGO_BIN if it is not on PATH, e.g.
REM      set CARGO_BIN=D:\rust\bin\cargo.exe
REM ---------------------------------------------------------------------------
setlocal

if not defined CARGO_BIN set "CARGO_BIN=cargo"

REM rc.exe lives under the Windows SDK's bin, in a version-named folder that is
REM not on PATH by default. Take the newest one that exists.
for /d %%D in ("C:\Program Files (x86)\Windows Kits\10\bin\10.*") do (
    if exist "%%D\x64\rc.exe" set "RC_DIR=%%D\x64"
)
if defined RC_DIR set "PATH=%RC_DIR%;%PATH%"

cd /d "%~dp0"

if /I "%~1"=="check" (
    echo [build] cargo check
    "%CARGO_BIN%" check --features embed
    goto :done
)
if /I "%~1"=="debug" (
    echo [build] debug ^(DirectML^)
    "%CARGO_BIN%" build
    goto :done
)
if /I "%~1"=="cuda" (
    echo [build] release ^(CUDA^)
    "%CARGO_BIN%" build --release --features embed,cuda
    goto :done
)

echo [build] release ^(DirectML, embedded bundle^)
"%CARGO_BIN%" build --release --features embed

:done
if errorlevel 1 (
    echo.
    echo [build] FAILED
    exit /b 1
)
echo.
echo [build] done -^> target\release\goblinscript.exe
endlocal
