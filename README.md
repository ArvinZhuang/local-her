# local-her


## Installation

```bash
conda create -n local-her python=3.10

pip3 install torch torchvision torchaudio
brew install ffmpeg
pip install mlx mlx-whisper mlx-lm

# Melo TTS and OpenVoice V2 converter
git clone git@github.com:myshell-ai/OpenVoice.git
cd OpenVoice
pip install -e .
cd ..
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

## OpenVoice V2 TTS converter
Download the checkpoint from [here](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip) and extract it to the checkpoints_v2 folder.

```bash
python3 convert_se.py \
--ckpt_converter checkpoints_v2/converter \
--reference xjp_audio.mp3 \
--target_dir xjp_audio
```

## Run

```bash
python3 her.py \
--target_se demo_speaker1/se.pth
```