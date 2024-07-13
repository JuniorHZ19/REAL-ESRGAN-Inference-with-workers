@echo off

:proceso
echo "Asegurese que dentro de inputs este la carpeta Test donde iran los frames de prueba"
echo -----------------------------------------------------------------------------------
set /p input=Ingrese carpeta que contiene las imagenes a escalar (debe estar en inputs): 
set contador=0
set procesos=
:Testeo
echo [1]  realesrgan-x4plus (Multiplica la resolución actual x4)
echo [2]  realesrgan-x2plus (Multiplica la resolución actual x2)
echo [3]  RealESRGAN_x4plus_anime_6B (Multiplica la resolución actual x4 para animes)
echo [4]  realesr-general-x4v3
echo [5]  realesr-animevideov3

set /p model_num=Ingrese numero de opcion de modelo: 

if "%model_num%"=="1" (
    set model_name=RealESRGAN_x4plus
)
if "%model_num%"=="2" (
    set model_name=RealESRGAN_x2plus
)
if "%model_num%"=="3" (
    set model_name=RealESRGAN_x4plus_anime_6B
)
if "%model_num%"=="4" (
    set model_name=realesr-general-x4v3
)
if "%model_num%"=="5" (
    set model_name=realesr-animevideov3
)

set /p scale=Ingrese la escala de multiplicacion: 
set /p tile=Ingrese el tamaño de tile: 

if %contador%==0 (
    python inference_realesrgan.py -i inputs/%input% -n %model_name% -s %scale% --suffix "" -o inputs/Test --tile %tile% --denoise_strength 1
) else (
    python inference_realesrgan.py -i inputs/Test -n %model_name% -s %scale% --suffix "" -o inputs/Test --tile %tile% --denoise_strength 1
)

set /a contador=%contador%+1

set procesos= %procesos% +%contador%) Modelo: %model_name%; Escala: %scale%; Tile: %tile%

cls
echo Procesos realizados:
echo.
echo %procesos%
echo.
echo.
goto Testeo

pause
