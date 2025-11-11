import argparse
import os
import sys

import numpy as np
import soundfile
from mir_eval.util import midi_to_hz

import onsets_and_frames.dataset as dataset_module
from onsets_and_frames import *


def load_and_process_audio(flac_path, sequence_length, device):

    random = np.random.RandomState(seed=42)

    audio, sr = soundfile.read(flac_path, dtype='int16')
    assert sr == SAMPLE_RATE

    audio = torch.ShortTensor(audio)

    if sequence_length is not None:
        audio_length = len(audio)
        step_begin = random.randint(audio_length - sequence_length) // HOP_LENGTH
        n_steps = sequence_length // HOP_LENGTH

        begin = step_begin * HOP_LENGTH
        end = begin + sequence_length

        audio = audio[begin:end].to(device)
    else:
        audio = audio.to(device)

    audio = audio.float().div_(32768.0)

    return audio


def transcribe(model, audio):

    mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

    predictions = {
            'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
            'offset': offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2])),
            'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
            'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
        }

    return predictions


def audio_sources(flac_paths, dataset, dataset_group, dataset_root, sequence_length, device):
    if dataset is not None:
        dataset_class = getattr(dataset_module, dataset)
        kwargs = {'sequence_length': None, 'device': device}
        if dataset_group is not None:
            kwargs['groups'] = [dataset_group]
        if dataset_root is not None:
            kwargs['path'] = dataset_root
        dataset_instance = dataset_class(**kwargs)
        for index in range(len(dataset_instance)):
            sample = dataset_instance[index]
            yield sample['path'], sample['audio']

    for flac_path in flac_paths:
        yield flac_path, load_and_process_audio(flac_path, sequence_length, device)


def transcribe_file(model_file, flac_paths, save_path, sequence_length,
                  onset_threshold, frame_threshold, device, dataset, dataset_group, dataset_root):

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    has_inputs = False
    for audio_path, audio in audio_sources(flac_paths, dataset, dataset_group, dataset_root, sequence_length, device):
        has_inputs = True
        print(f'Processing {audio_path}...', file=sys.stderr)
        predictions = transcribe(model, audio)

        p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'], onset_threshold, frame_threshold)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        os.makedirs(save_path, exist_ok=True)
        base_name = os.path.basename(audio_path)
        pred_path = os.path.join(save_path, base_name + '.pred.png')
        save_pianoroll(pred_path, predictions['onset'], predictions['frame'])
        midi_path = os.path.join(save_path, base_name + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)

    if not has_inputs:
        raise ValueError('No audio inputs provided. Supply file paths or --dataset.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('flac_paths', type=str, nargs='*')
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', default=None, help='Optional dataset class name (e.g., BabySlakhDrums)')
    parser.add_argument('--dataset-group', default=None)
    parser.add_argument('--dataset-root', default=None)

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))
