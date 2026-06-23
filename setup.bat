@echo off
echo =====================================================
echo Tide Engine Dependency Setup
echo =====================================================

:: Check if CMake is installed
where cmake >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake could not be found! Please install CMake and ensure it is in your PATH.
    pause
    exit /b 1
)

echo.
echo [1/1] Building GLFW (Debug and Release)...
cd Dependency\glfw
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency\glfw folder not found! Did you forget to clone with --recursive?
    pause
    exit /b 1
)

echo Running CMake Configure...
cmake -B build
if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configure failed!
    pause
    exit /b 1
)

echo Compiling Debug configuration...
cmake --build build --config Debug
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Debug build failed!
    pause
    exit /b 1
)

echo Compiling Release configuration...
cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Release build failed!
    pause
    exit /b 1
)

cd ..\..

echo.
echo =====================================================
echo SUCCESS! All dependencies are built.
echo You can now open Tide Engine.sln in Visual Studio 2022.
echo =====================================================
pause
