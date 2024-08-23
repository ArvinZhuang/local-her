import mlx_whisper
import sounddevice as sd
import numpy as np
from pynput import keyboard
import torch
from vad import VADIterator, int2float
from melo.api import TTS
from rich.console import Console
from mlx_lm import load, stream_generate, generate

console = Console()
key_pressed = False
fs = 44100  # Sample rate
recording = []  # List to store audio chunks
thresh = 0.3
sample_rate = 16000
min_silence_ms = 1000
min_speech_ms = 500
max_speech_ms = float("inf")
speech_pad_ms = 30
stt_model = 'mlx-community/whisper-large-v3-mlx-4bit'


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
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer


def main():
    tts_model = TTS(language='ZH', device='mps')
    speaker_ids = tts_model.hps.data.spk2id
    llm_model, tokenizer = load('mlx-community/Qwen2-7B-Instruct-4bit')

    # warm up
    _ = mlx_whisper.transcribe(np.array([0] * 512), path_or_hf_repo=stt_model, language='zh', fp16=True)["text"]
    test_audio = tts_model.tts_to_file('让我们开始吧!', speaker_ids['ZH'], quiet=True)
    sd.play(test_audio, fs)

    chat = Chat(10)
    chat.init_chat({"role": 'system', "content": '你是一个中文助手, 你会可以聊天和回答问题, 你的回答必须尽可能简短.'})

    # model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")

    # iterator = VADIterator(
    #     model,
    #     threshold=thresh,
    #     sampling_rate=sample_rate,
    #     min_silence_duration_ms=min_silence_ms,
    #     speech_pad_ms=speech_pad_ms,
    # )

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
            if key == keyboard.Key.cmd_r and not key_pressed:  # Change 'a' to the key you want to monitor
                key_pressed = True
                recording = []  # Start with a new recording
                stream.start()  # Start the audio stream
        except AttributeError:
            pass

    def on_release(key):
        global key_pressed
        try:
            if key == keyboard.Key.cmd_r:  # Change 'a' to the key you want to monitor
                key_pressed = False
                stream.stop()  # Stop the audio stream

                if recording:  # Save the recording to a WAV file
                    audio_int16 = np.array(recording).flatten()
                    audio_float32 = int2float(audio_int16)
                    user_text = mlx_whisper.transcribe(audio_float32,
                                                       path_or_hf_repo=stt_model,
                                                       language='zh',
                                                       fp16=True)["text"].strip()

                    torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
                    torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.

                    console.print(f"[yellow]USER: {user_text}")
                    chat.append({"role": 'user', "content": user_text})

                    prompt = tokenizer.apply_chat_template(chat.to_list(), tokenize=False, add_generation_prompt=True)

                    response = generate(llm_model, tokenizer, prompt=prompt, verbose=False)

                    console.print(f"[green]HER: {response}")

                    chat.append({"role": 'assistant', "content": response})

                    audio_chunk = tts_model.tts_to_file(response, speaker_ids['ZH'], quiet=True)
                    sd.play(audio_chunk, fs)

        except AttributeError:
            pass

    # Start a keyboard listener
    print('Press "cmd_r" to start')
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    main()
