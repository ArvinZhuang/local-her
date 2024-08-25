# local-her
Speech-to-speech local AI assistant that is optimized for Mac silicon devices. (Inspired by Huggingface [speech-to-speech](https://github.com/huggingface/speech-to-speech) project)

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
## Run

```bash
python3 her.py
```

## OpenVoice V2 TTS converter
Convert base specker voice to any other speaker.
Download the checkpoint from [here](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip) and extract it to the checkpoints_v2 folder.

```bash
python3 convert_se.py \
--ckpt_converter checkpoints_v2/converter \
--reference OpenVoice/resources/demo_speaker2.mp3 \
--target_dir demo_speaker2

# then
python3 her.py \
--target_se demo_speaker2/se.pth
```
