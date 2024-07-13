<p align="center">
  <img src="assets/realesrgan_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">

👀[**Demos**](#-demos-videos) **|** 🚩[**Updates**](#-updates) **|** ⚡[**Usage**](#-quick-inference) **|** 🏰[**Model Zoo**](docs/model_zoo.md) **|** 🔧[Install](#-dependencies-and-installation)  **|** 💻[Train](docs/Training.md) **|** ❓[FAQ](docs/FAQ.md) **|** 🎨[Contribution](docs/CONTRIBUTING.md)

[![download](https://img.shields.io/github/downloads/xinntao/Real-ESRGAN/total.svg)](https://github.com/xinntao/Real-ESRGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/realesrgan)](https://pypi.org/project/realesrgan/)
[![Open issue](https://img.shields.io/github/issues/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/Real-ESRGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/Real-ESRGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/publish-pip.yml)

</div>

🔥 **AnimeVideo-v3 model (动漫视频小模型)**. Please see [[*anime video models*](docs/anime_video_model.md)] and [[*comparisons*](docs/anime_comparisons.md)]<br>
🔥 **RealESRGAN_x4plus_anime_6B** for anime images **(动漫插图模型)**. Please see [[*anime_model*](docs/anime_model.md)]

<!-- 1. You can try in our website: [ARC Demo](https://arc.tencent.com/en/ai-demos/imgRestore) (now only support RealESRGAN_x4plus_anime_6B) -->
1. :boom: **Update** online Replicate demo: [![Replicate](https://img.shields.io/static/v1?label=Demo&message=Replicate&color=blue)](https://replicate.com/xinntao/realesrgan)
1. Online Colab demo for Real-ESRGAN: [![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)](https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing) **|** Online Colab demo for for Real-ESRGAN (**anime videos**): [![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)](https://colab.research.google.com/drive/1yNl9ORUxxlL4N0keJa2SEPB61imPQd1B?usp=sharing)
1. Portable [Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip) / [Linux](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip) / [MacOS](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip) **executable files for Intel/AMD/Nvidia GPU**. You can find more information [here](#portable-executable-files-ncnn). The ncnn implementation is in [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
<!-- 1. You can watch enhanced animations in [Tencent Video](https://v.qq.com/s/topic/v_child/render/fC4iyCAM.html). 欢迎观看[腾讯视频动漫修复](https://v.qq.com/s/topic/v_child/render/fC4iyCAM.html) -->

Real-ESRGAN aims at developing **Practical Algorithms for General Image/Video Restoration**.<br>
We extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data.

🌌 Thanks for your valuable feedbacks/suggestions. All the feedbacks are updated in [feedback.md](docs/feedback.md).

---

If Real-ESRGAN is helpful, please help to ⭐ this repo or recommend it to your friends 😊 <br>
Other recommended projects:<br>
▶️ [GFPGAN](https://github.com/TencentARC/GFPGAN): A practical algorithm for real-world face restoration <br>
▶️ [BasicSR](https://github.com/xinntao/BasicSR): An open-source image and video restoration toolbox<br>
▶️ [facexlib](https://github.com/xinntao/facexlib): A collection that provides useful face-relation functions.<br>
▶️ [HandyView](https://github.com/xinntao/HandyView): A PyQt5-based image viewer that is handy for view and comparison <br>
▶️ [HandyFigure](https://github.com/xinntao/HandyFigure): Open source of paper figures <br>

---

### 📖 Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

> [[Paper](https://arxiv.org/abs/2107.10833)] &emsp; [[YouTube Video](https://www.youtube.com/watch?v=fxHWoDSSvSc)] &emsp; [[B站讲解](https://www.bilibili.com/video/BV1H34y1m7sS/)] &emsp; [[Poster](https://xinntao.github.io/projects/RealESRGAN_src/RealESRGAN_poster.pdf)] &emsp; [[PPT slides](https://docs.google.com/presentation/d/1QtW6Iy8rm8rGLsJ0Ldti6kP-7Qyzy6XL/edit?usp=sharing&ouid=109799856763657548160&rtpof=true&sd=true)]<br>
> [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> [Tencent ARC Lab](https://arc.tencent.com/en/ai-demos/imgRestore); Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

<p align="center">
  <img src="assets/teaser.jpg">
</p>

---

<!---------------------------------- Updates --------------------------->
## 🚩 Updates

- ✅ Add the **realesr-general-x4v3** model - a tiny small model for general scenes. It also supports the **-dn** option to balance the noise (avoiding over-smooth results). **-dn** is short for denoising strength.
- ✅ Update the **RealESRGAN AnimeVideo-v3** model. Please see [anime video models](docs/anime_video_model.md) and [comparisons](docs/anime_comparisons.md) for more details.
- ✅ Add small models for anime videos. More details are in [anime video models](docs/anime_video_model.md).
- ✅ Add the ncnn implementation [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan).
- ✅ Add [*RealESRGAN_x4plus_anime_6B.pth*](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth), which is optimized for **anime** images with much smaller model size. More details and comparisons with [waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan) are in [**anime_model.md**](docs/anime_model.md)
- ✅ Support finetuning on your own data or paired data (*i.e.*, finetuning ESRGAN). See [here](docs/Training.md#Finetune-Real-ESRGAN-on-your-own-dataset)
- ✅ Integrate [GFPGAN](https://github.com/TencentARC/GFPGAN) to support **face enhancement**.
- ✅ Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/Real-ESRGAN). Thanks [@AK391](https://github.com/AK391)
- ✅ Support arbitrary scale with `--outscale` (It actually further resizes outputs with `LANCZOS4`). Add *RealESRGAN_x2plus.pth* model.
- ✅ [The inference code](inference_realesrgan.py) supports: 1) **tile** options; 2) images with **alpha channel**; 3) **gray** images; 4) **16-bit** images.
- ✅ The training codes have been released. A detailed guide can be found in [Training.md](docs/Training.md).

---

<!---------------------------------- Demo videos --------------------------->
## 👀 Demos Videos

#### Bilibili

- [大闹天宫片段](https://www.bilibili.com/video/BV1ja41117zb)
- [Anime dance cut 动漫魔性舞蹈](https://www.bilibili.com/video/BV1wY4y1L7hT/)
- [海贼王片段](https://www.bilibili.com/video/BV1i3411L7Gy/)

#### YouTube

## 🔧 Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/JuniorHZ19/REAL-ESRGAN-Inference-with-workers.git
    cd REAL-ESRGAN-Inference-with-workers
    ```

1. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr
    # facexlib and gfpgan are for face enhancement
    pip install facexlib
    pip install gfpgan
    pip install -r requirements.txt
    python setup.py develop
    ```

---

## ⚡ Quick Inference

There are usually three ways to inference Real-ESRGAN.

1. [Online inference](#online-inference)
1. [Portable executable files (NCNN)](#portable-executable-files-ncnn)
1. [Python script](#python-script)


We can use five models:

1. realesrgan-x4plus  (default)
2. realesrnet-x4plus
3. realesrgan-x4plus-anime (optimized for anime images, small model size)
4. realesr-animevideov3 (animation video)

You can use the `-n` argument for other models, for example, `./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n realesrnet-x4plus`



### Python script

#### Usage of python script

1. You can use X4 model for **arbitrary output size** with the argument `outscale`. The program will further perform cheap resize operation after the Real-ESRGAN output.

```console
Usage Original inference: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
  --workers            The number of workers can depend on your GPU; more workers require more GPU memory. .Default:1 (New implementation)
```

#### Inference general images

Download pre-trained models: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```

Inference Normal Way!

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance --workers
Results are in the `results` folder

```
Inference with Workers!

```bash
python inference_realesrgan_video-multiproceso.py -i 'inputs/frames' -n RealESRGAN_x4plus -o 'inputs/frames' --workers 2
```
Results and the input must be the same place in the `inputs/frames` folder , you can use the original arguments except face_enhance

#### Inference anime images

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>

Pre-trained models: [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)<br>
 More details and comparisons with [waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan) are in [**anime_model.md**](docs/anime_model.md)

```bash
# download model
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights
# inference
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs
```

Results are in the `results` folder

---

## BibTeX

    @InProceedings{wang2021realesrgan,
        author Original    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan}
        author of the modification ={Junior ,Huari Z.}
        title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
        date      = {2021}
    }

## 📧 Contact

If you have any question, please email `xintao.wang@outlook.com`,`xintaowang@tencent.com`. for the original autor  or `testjhz19@gmail.com` for the author of the modification.

