import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from argparse import ArgumentParser


def main():
    argparse = ArgumentParser()
    argparse.add_argument('--ckpt_converter', type=str, help='path to the converter checkpoint')
    argparse.add_argument('--reference', type=str, help='path to the reference in mp3')
    argparse.add_argument('--device', type=str, default='mps', help='device to run the model on')
    argparse.add_argument('--target_dir', type=str, help='output directory')
    args = argparse.parse_args()

    ckpt_converter = args.ckpt_converter

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=args.device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    reference_speaker = args.reference  # This is the voice you want to clone
    target_se, audio_name = se_extractor.get_se(reference_speaker,
                                                tone_color_converter,
                                                target_dir=args.target_dir,
                                                vad=False)
    torch.save(target_se, f'{args.target_dir}/se.pth')


if __name__ == '__main__':
    main()