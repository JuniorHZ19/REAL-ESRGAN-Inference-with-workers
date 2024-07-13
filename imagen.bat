@echo off

:proceso

set /p input="Ingrese nombre imagen de entrada ,que esta en la ruta inputs/:"

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

set /p subfijo="Ingrese el subfijo que se ponndra al output:"

set /p tile="Ingrese el size tile:"

python inference_realesrgan.py -i inputs/%input% -n %model_name% -s %scale% --suffix %subfijo% --tile %tile%

cls
goto proceso

