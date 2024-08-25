import mlx_whisper
import sounddevice as sd
import numpy as np
from pynput import keyboard
import torch
from melo.api import TTS
from rich.console import Console
from mlx_lm import load, stream_generate, generate
import threading
import nltk
import re
import time
from openvoice.api import ToneColorConverter
from transformers import AutoProcessor, AutoModel, BarkModel
import soundfile
import librosa
from argparse import ArgumentParser

console = Console()
key_pressed = False
recording = []  # List to store audio chunks


def int2float(sound):
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


class Chat:
    """
    Handles the chat using to avoid OOM issues.
    """

    def __init__(self, size):
        self.size = size
        self.init_chat_message = None
        # maxlen is necessary pair, since a each new step we add an prompt and assitant answer
        self.buffer = []

    def append(self, item):
        self.buffer.append(item)
        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)

    def init_chat(self, init_chat_message):
        self.init_chat_message = init_chat_message

    def to_list(self):
        if self.init_chat_message:
            return self.init_chat_message + self.buffer
        else:
            return self.buffer


class ToneConverter:
    def __init__(self, model, source_se, target_se, orig_sr, target_sr):
        self.model = model
        self.source_se = source_se
        self.target_se = target_se
        self.orig_sr = orig_sr
        self.target_sr = target_sr

    def convert(self, audio):
        audio = librosa.resample(audio,
                                 orig_sr=self.orig_sr,
                                 target_sr=self.target_sr)
        audio = self.model.convert_audio(
            audio=audio,
            src_se=self.source_se,
            tgt_se=self.target_se)
        return audio


def main():
    argparse = ArgumentParser()
    argparse.add_argument('--verbose', action='store_true')
    argparse.add_argument('--stt_model', type=str, default='mlx-community/whisper-large-v3-mlx-4bit')
    argparse.add_argument('--llm_model', type=str, default='mlx-community/gemma-2-2b-it-8bit')
    argparse.add_argument('--tts_language', type=str, default='ZH')
    argparse.add_argument('--ckpt_converter', type=str, default='checkpoints_v2/converter')
    argparse.add_argument('--source_se', type=str, default='checkpoints_v2/base_speakers/ses/zh.pth')
    argparse.add_argument('--target_se', type=str, default=None)
    argparse.add_argument('--lm_max_tokens', type=int, default=100)
    argparse.add_argument('--sample_rate', type=int, default=16000)
    argparse.add_argument('--device', type=str, default='mps')

    args = argparse.parse_args()

    verbose = args.verbose
    stt_model = args.stt_model
    llm_model, tokenizer = load(args.llm_model)
    tts_model = TTS(language=args.tts_language, device=args.device)
    speaker_ids = tts_model.hps.data.spk2id
    ckpt_converter = args.ckpt_converter

    if args.target_se is not None:
        cv_model = ToneColorConverter(f'{ckpt_converter}/config.json', device=args.device)
        cv_model.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        tone_color_converter = ToneConverter(
            model=cv_model,
            source_se=torch.load(args.source_se, map_location=args.device),
            target_se=torch.load(args.target_se, map_location=args.device),
            orig_sr=tts_model.hps.data.sampling_rate,
            target_sr=cv_model.hps.data.sampling_rate
        )
    else:
        tone_color_converter = None

    lm_max_tokens = args.lm_max_tokens
    sample_rate = args.sample_rate

    # warm up
    _ = mlx_whisper.transcribe(np.array([0] * 512), path_or_hf_repo=stt_model, language='zh', fp16=True)["text"]
    test_audio = tts_model.tts_to_file('让我们开始吧!', speaker_ids['ZH'], quiet=True)
    if tone_color_converter:
        test_audio = tone_color_converter.convert(test_audio)
        sd.play(test_audio, tone_color_converter.target_sr)
    else:
        sd.play(test_audio, tts_model.hps.data.sampling_rate)

    chat = Chat(5)
    chat.init_chat(
        [{"role": 'user', "content": '你是一个中文助手.用简短的话回答问题.'},
         {"role": 'assistant', "content": '好的没问题!我一定会回答所有你的问题!并且用简短的语言.'}])

    # Create an audio stream
    def callback(indata, frames, time, status):
        """This callback function will be called during recording."""
        if key_pressed:
            recording.append(indata.copy())

    stream = sd.InputStream(callback=callback,
                            channels=1,
                            dtype="int16",
                            samplerate=sample_rate,
                            blocksize=512)

    def on_press(key):
        global key_pressed, recording
        try:
            if key == keyboard.Key.cmd_r and not key_pressed:  # right size cmd
                print("Listening...")
                sd.stop()
                key_pressed = True
                recording = []  # Start with a new recording
                stream.start()  # Start the audio stream

        except AttributeError:
            pass

    def on_release(key):
        global key_pressed
        try:
            if key == keyboard.Key.cmd_r:  # right size cmd
                key_pressed = False
                stream.stop()  # Stop the audio stream

                if recording:
                    # audio length in seconds
                    audio_int16 = np.array(recording).flatten()
                    audio_length = len(audio_int16) / sample_rate

                    if audio_length < 1:
                        console.print("[red]No audio detected.")
                        return

                    audio_float32 = int2float(audio_int16)
                    user_text = mlx_whisper.transcribe(audio_float32,
                                                       path_or_hf_repo=stt_model,
                                                       fp16=True)["text"].strip()

                    console.print(f"[yellow]USER: {user_text}")
                    chat.append({"role": 'user', "content": user_text})
                    prompt = tokenizer.apply_chat_template(chat.to_list(), tokenize=False, add_generation_prompt=True)

                    response = generate(llm_model, tokenizer, prompt=prompt, verbose=verbose, max_tokens=lm_max_tokens)
                    response = response.replace('<end_of_turn>', '')
                    response = response.replace('\n', ' ')

                    console.print(f"[green]HER: {response}")
                    chat.append({"role": 'assistant', "content": response})

                    sentences = tts_model.split_sentences_into_pieces(response, args.tts_language, quiet=True)
                    for sentence in sentences:
                        audio = tts_model.tts_to_file(sentence, speaker_ids[args.tts_language], quiet=True)
                        sd.wait()
                        if tone_color_converter:
                            audio = tone_color_converter.convert(audio)
                            sd.play(audio, tone_color_converter.target_sr)
                        else:
                            sd.play(audio, tts_model.hps.data.sampling_rate)
                else:
                    console.print("[red]No audio detected.")

        except AttributeError:
            pass

    # Start a keyboard listener
    print('Press "cmd_r" to start')
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    main()
