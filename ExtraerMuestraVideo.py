import subprocess
import os

#Sacamos el minuto exacto de inicio y final del video

minutoInicio=os.environ.get("inicio")
minutoFinal=os.environ.get("final")

print ("minutos",minutoInicio,minutoFinal)
# Comando Para hacer muestra
ffmpeg_command = [
    "ffmpeg",
    "-i",'inputs/data_dst.mp4',
     '-ss',  
       minutoInicio,
      '-to',
        minutoFinal,
      'inputs/data_dst_muestra.mp4'
]

try:
    subprocess.run(ffmpeg_command, check=True)
    print("Se escalo el video satisfactoriamente.")
except subprocess.CalledProcessError as e:
    print(f"Error al escalar el video: {e}")
