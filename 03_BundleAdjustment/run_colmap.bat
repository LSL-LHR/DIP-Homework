@echo off
setlocal enabledelayedexpansion

REM 设置路径
set "COLMAP_EXE=D:\Program Files\colmap-x64-windows-cuda\bin\colmap.exe"
set "PROJECT_DIR=%~dp0data"
set "IMAGE_DIR=%PROJECT_DIR%\images"
set "WORK_DIR=%PROJECT_DIR%\colmap"

echo ========================================
echo COLMAP 3D Reconstruction Pipeline
echo ========================================

REM 创建工作目录
if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"
if not exist "%WORK_DIR%\sparse" mkdir "%WORK_DIR%\sparse"
if not exist "%WORK_DIR%\dense" mkdir "%WORK_DIR%\dense"

echo [Step 1/6] Feature Extraction...
call "%COLMAP_EXE%" feature_extractor ^
    --database_path "%WORK_DIR%\database.db" ^
    --image_path "%IMAGE_DIR%" ^
    --ImageReader.camera_model PINHOLE ^
    --ImageReader.single_camera 1

echo [Step 2/6] Feature Matching...
call "%COLMAP_EXE%" exhaustive_matcher ^
    --database_path "%WORK_DIR%\database.db"

echo [Step 3/6] Sparse Reconstruction (Bundle Adjustment)...
call "%COLMAP_EXE%" mapper ^
    --database_path "%WORK_DIR%\database.db" ^
    --image_path "%IMAGE_DIR%" ^
    --output_path "%WORK_DIR%\sparse"

echo [Step 4/6] Image Undistortion...
call "%COLMAP_EXE%" image_undistorter ^
    --image_path "%IMAGE_DIR%" ^
    --input_path "%WORK_DIR%\sparse\0" ^
    --output_path "%WORK_DIR%\dense"

echo [Step 5/6] Dense Reconstruction (Patch Match Stereo)...
call "%COLMAP_EXE%" patch_match_stereo ^
    --workspace_path "%WORK_DIR%\dense"

echo [Step 6/6] Stereo Fusion...
call "%COLMAP_EXE%" stereo_fusion ^
    --workspace_path "%WORK_DIR%\dense" ^
    --output_path "%WORK_DIR%\dense\fused.ply"

echo ========================================
echo COLMAP Pipeline Complete!
echo Results:
echo   Sparse model: %WORK_DIR%\sparse\0
echo   Dense point cloud: %WORK_DIR%\dense\fused.ply
echo ========================================
pause