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


ffmpeg -framerate %framerate% -i "%BASE_DIR%inputs\Test\frame_%%04d.png" -c:v libx264  -crf 16 -preset slow -r %rate%  "%BASE_DIR%inputs\Test\temp.mp4"

ffmpeg -i "%BASE_DIR%\inputs\Test\temp.mp4" -i %BASE_DIR%inputs\video\%input% -c:v copy -c:a aac -strict experimental results/%ouput_sf%.mp4

del "%BASE_DIR%inputs\Test\temp.mp4"

"video creado en results/:  %ouput_sf%_escalado.mp4"

pause
