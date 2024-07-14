@echo off

echo No olvidar que la carpeta inputs/test , tenga los frames ya escalados a nuestra conveniencia":
echo ---------------------------------------------------------------------------------------------------------
set /p input="Poner nombre orignal del archivo en inputs/video :"
set /p ouput_sf="Poner el nombre del archivo final que se guardara en resuls/:"
goto Unir_frames

echo "Video creado exitosamente"


:Unir_frames

set "BASE_DIR=%~dp0"

echo  -----------------------------------------------------------------------------------
echo "Asegurese de que los frames esten escalados en la carpeta inputs/Test"
echo  -----------------------------------------------------------------------------------



set /p framerate="espesfia la tasa original del video:"
set /p rate="espesfia la tasa que quiere que este creado el video final:"



ffmpeg -framerate %framerate% -i  "%BASE_DIR%inputs\frames\frame_%%04d.png" -i %BASE_DIR%inputs\video\%input% -map 0:v -map 1:a -pix_fmt yuv420p -c:a mp3 -crf 16 -preset slow -r %rate%  -strict experimental results/%ouput_sf%.mp4


"video creado en results/:  %ouput_sf%_escalado.mp4"

pause
