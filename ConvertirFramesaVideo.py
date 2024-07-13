import subprocess
import os



# Comando Para unir frames a video
ffmpeg_command = [
    "ffmpeg",
    "-framerate",
    "30",
    "-i",'results/ScaledFrames/frame_%04d_out.png',
    "-i",
    "inputs/data_dst.mp4",
     
     '-c:v',  
       'libx264',
      '-c:a',
        'mp3',
       '-strict',
      'experimental',
      '-vf',
       'fps=30',
      '-shortest',
      'results/result.mp4'
]

try:
    subprocess.run(ffmpeg_command, check=True)
    print("Se escalo el video satisfactoriamente.")
except subprocess.CalledProcessError as e:
    print(f"Error al escalar el video: {e}")
