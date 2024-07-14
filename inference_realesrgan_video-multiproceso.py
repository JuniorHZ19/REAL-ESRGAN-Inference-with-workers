import argparse
import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path as osp
from tqdm import tqdm
from queue import Queue
from realesrgan import RealESRGANer
import threading
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import psutil
from typing import Any, List, Callable
import numpy as np
try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg

THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
UPSCALER = None
def create_frames(args):
   delete_frames(args)
   print(f"Extracting frames ....")
   cmd = [
       'ffmpeg',
       '-i',str(args.input),
       '-qscale:v','1',
       '-qmin','1',
       '-qmax','1',
       '-vsync','0',
       str(args.output_frames)+'/frame_%04d.png']

   process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   stdout, stderr = process.communicate()
   if process.returncode != 0:
    print(stderr)
    raise RuntimeError(stderr)
   else:
    frame_count = len(os.listdir(args.output_frames))
    print(f"Done, Extracted {frame_count} Frames")

def delete_frames(args):
    for item in os.listdir(args.output_frames):
        item_path = os.path.join(args.output_frames, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Eliminar directorios y su contenido
        else:
            os.remove(item_path)  # Eliminar archivos
    print(f"Deleted frames ....")

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret



def create_video(args):
   print("Creating Video......")
   info=get_video_meta_info(args.input)
   cmd = [
       'ffmpeg',
       '-framerate',str(info['fps']),
       '-i',str(args.output_frames)+'/frame_%04d.png',
       '-i',str(args.input),
       '-map','0:v',
       '-map','1:a',
       '-c:v','libx264',
       '-crf','16',
       '-preset','veryslow',
       '-pix_fmt', 'yuv420p',
       '-c:a','mp3',
       '-r',str(info['fps']),
        "results/"+str(args.output_video)
        ]
   process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   stdout, stderr = process.communicate()
   if process.returncode != 0:
    print(stderr)
    raise RuntimeError(stderr)
   else:
    print("Done Recreating Video")


def getPaths(carpeta):
    paths = []
    for root, dirs, files in os.walk(carpeta):
        for file in files:
            paths.append(os.path.join(root, file))

    return paths


def divide_chunks(lst, n):
    """Divide una lista en chunks de tamaño n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def multi_process_frame(args, temp_frame_paths, process_frames, update):
    # Calcula el tamaño de cada chunk
    chunk_size = max(len(temp_frame_paths) // args.workers, 1)
    chunks = list(divide_chunks(temp_frame_paths, chunk_size))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for chunk in chunks:
            future = executor.submit(process_frames, args, chunk, update)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing frame: {e}")

def process_video(args,frame_paths, process_frames) :
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        multi_process_frame(args,frame_paths, process_frames, lambda: update_progress(progress))


def update_progress(progress):
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
    progress.set_postfix({
        'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',

    })
    progress.refresh()
    progress.update(1)






def upscale_frame(args,path,upsampler):

    imgname, extension = os.path.splitext(os.path.basename(path))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
    else:
            img_mode = None

    try:
        with THREAD_SEMAPHORE:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
    except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number or reduce the number of workers.')
    else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if args.suffix == '':
                save_path = os.path.join(args.output_frames, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output_frames, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)



def upsacle_frames(args,temp_frame_paths,update):

     upsampler=getModel(args)

     for temp_frame_path in temp_frame_paths:

      upscale_frame(args,temp_frame_path,upsampler)

      if update:
            update()

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret


def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path

def getModel(args):
    global UPSCALER
    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    with THREAD_LOCK:

     if UPSCALER is None:
      UPSCALER = RealESRGANer(
      scale=netscale,
      model_path=model_path,
      dni_weight=dni_weight,
      model=model,
      tile=args.tile,
      tile_pad=args.tile_pad,
      pre_pad=args.pre_pad,
      half=not args.fp32,
      gpu_id=args.gpu_id)

    return UPSCALER


def main():

    """
    Crear carpeta donde iran los frames:
    """
    carpeta_input='inputs'
    carpeta_frames='inputs/frames'
    carpeta_results='results'
    if not os.path.exists(carpeta_frames):
     os.makedirs(carpeta_frames, exist_ok=True)
    elif not os.path.exists(carpeta_input):
     os.makedirs(carpeta_input, exist_ok=True)
    elif not os.path.exists(carpeta_results):
     os.makedirs(carpeta_results, exist_ok=True)

    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output_frames', type=str, default='inputs/frames', help='Output folder')
    parser.add_argument('--output_video', type=str, help='Output folder')
    parser.add_argument('--workers', type=int, default=1, help='Num of worker to multiprocess')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    parser.add_argument('--keep_frames',action='store_true',default=False, help='Keep the frames so you can rescale them later.')
    args = parser.parse_args()


    extension = os.path.splitext(args.input)[1].lower()

    if extension == '.mp4' or extension == '.avi' or extension == '.mov' or extension == '.mkv' or extension == '.flv' or extension == '.wmv':
      create_frames(args)
      paths=sorted(getPaths(args.output_frames))

      process_video(args,paths,upsacle_frames)
      create_video(args)

    else:
      paths = sorted(getPaths(args.input))
      process_video(args,paths, upsacle_frames)


    if(args.keep_frames==False):
      delete_frames(args)

if __name__ == '__main__':
    main()



