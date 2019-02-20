import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from datasets import audio
from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize
import shutil
from nnmnkwii.preprocessing import interp1d
import soundfile as sf
from datasets import utils

def build_from_path(hparams, input_dir, cmp_dir, linear_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited

	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []

	lpc_input_dir = os.path.join(input_dir, 'inputs')
	if(not os.path.exists(lpc_input_dir)):
		os.makedirs(lpc_input_dir)

	with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			basename = parts[0]
			wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
			if(not os.path.exists(wav_path)):
				continue;
			text = parts[1]
			futures.append(executor.submit(partial(_process_utterance, lpc_input_dir, cmp_dir, linear_dir, basename, wav_path, text, hparams)))

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(input_dir, cmp_dir, linear_dir, basename, wav_path, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- basename:
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	filename = os.path.basename(wav_path).split(".")[0]

	input_path = os.path.join(input_dir, filename)
	output_path = os.path.join(cmp_dir, "%s.cmp" % filename)
	# get inputs
	os.system("sox %s -t sw -r 16000 -c 1 %s" % (wav_path, input_path))
	os.system("dump_data_t -test %s %s" % (input_path, output_path))

	cep = utils.read_cep_pitch(output_path)
	cmp_frames = len(cep)

	wav = audio.load_wav(wav_path, hparams.sample_rate)
	linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
	linear_filename = 'linear-{}.npy'.format(basename)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)
	# Return a tuple describing this training example
	return (os.path.basename(output_path), linear_filename, cmp_frames, text)
