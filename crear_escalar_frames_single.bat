@echo off

:proceso

echo "Asegurese dentro de inputs este la carepta Test es donde ira los frames de prueba"
echo  -----------------------------------------------------------------------------------

set /p input="Ingrese nombre del video a testear este video debe tar en inputs/video o poner (n) si ya tenemos las imagens en test:"
set contador=0
set procesos=

set "BASE_DIR=%~dp0"


if  not "%input%"== "n" (

   ffmpeg -i inputs/video/%input%  "%BASE_DIR%\inputs\Test\frame_%%04d.png"
   cls
)

:Testeo

echo [1]  realesrgan-x4plus( Multiplica la resulcion actual x4)
echo [2]  realesrgan-x2plus( Multiplica la resulcion actual x2)
echo [3]  RealESRGAN_x4plus_anime_6B(Multiplica la resulcion actual x4 para animes)
echo [4]  realesr-general-x4v3
echo [5] realesr-animevideov3

set /p model_num=Ingrese numero opcion de modelo:
                

if "%model_num%"== "1" (
    set model_name="RealESRGAN_x4plus"
 
)
if "%model_num%"== "2" (
    set model_name="RealESRGAN_x2plus"
    
)
if "%model_num%"== "3" (
    set model_name="RealESRGAN_x4plus_anime_6B"
   
)
if "%model_num%"== "4" (
    set model_name="realesr-general-x4v3"
  
)
if "%model_num%"== "5" (
    set model_name="realesr-animevideov3"
    
)

set /p scale="Ingrese la escala de mulitplicacion:"


set /p tile="Ingrese el size tile:"

set /p worker="Ingrese numero de trabajos al a vez como maximo:"

set /a contador=%contador%+1

set procesos= %procesos% %contador%) Modelo: %model_name%; Escala: %scale%; Tile: %tile%

python inference_realesrgan_video-multiproceso.py -i inputs/Test -n %model_name% -s %scale% --suffix "" -o inputs/Test --tile %tile%  --denoise_strength 1 --workers 1 
pause
cls

echo Procesos realizados:
echo.
echo %procesos%
echo.
echo.

goto Testeo

pause
