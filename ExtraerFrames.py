import subprocess
import os
# Ruta al archivo de video de entrada
input_video = "inputs/data_dst.mp4"

# Ruta donde se guardarán los fotogramas
output_folder = "results/frames"

# Comando de FFmpeg para extraer los fotogramas
ffmpeg_command = [
    "ffmpeg",
    "-i", input_video,
    os.path.join(output_folder, "frame_%04d.png")
]

try:
    subprocess.run(ffmpeg_command, check=True)
    print("Fotogramas extraídos con éxito.")
except subprocess.CalledProcessError as e:
    print(f"Error al extraer fotogramas: {e}")
