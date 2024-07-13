import subprocess
import os



# Comando Para hacer muestra
ffmpeg_command = [
   "ffmpeg",
    "-i",'results/ScaledFrames/frame_%04d_out.png',
    "-i",
    "inputs/data_dst_muestra.mp4",
     
     '-c:v',  
       'libx264',
      '-c:a',
        'mp3',
     
      'results/resultmuestra.mp4'
]

try:
    subprocess.run(ffmpeg_command, check=True)
    print("Se escalo el video satisfactoriamente.")
except subprocess.CalledProcessError as e:
    print(f"Error al escalar el video: {e}")
