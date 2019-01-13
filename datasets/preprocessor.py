import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from datasets import audio
from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize
import shutil

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
	lf0_dir = os.path.join(input_dir, "lf0")
	mgc_dir = os.path.join(input_dir, "mgc")
	bap_dir = os.path.join(input_dir, "bap")
	for dire in [lf0_dir, mgc_dir, bap_dir]:
		if (os.path.exists(dire)):
			shutil.rmtree(dire)
		os.makedirs(dire, exist_ok=False)
	with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			basename = parts[0]
			wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
			text = parts[1]
			futures.append(executor.submit(partial(_process_utterance, lf0_dir, mgc_dir, bap_dir, cmp_dir, linear_dir, basename, wav_path, text, hparams)))

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(lf0_dir, mgc_dir, bap_dir, cmp_dir, linear_dir, basename, wav_path, text, hparams):
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
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#Pre-emphasize
	wav = audio.preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#Assert all audio is in [-1, 1]
	if (wav > 1.).any() or (wav < -1.).any():
		raise RuntimeError('wav has invalid value: {}'.format(wav))

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#Mu-law quantize
	if is_mulaw_quantize(hparams.input_type):
		#[0, quantize_channels)
		out = mulaw_quantize(wav, hparams.quantize_channels)

		#Trim silences
		start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
		wav = wav[start: end]
		out = out[start: end]

		constant_values = mulaw_quantize(0, hparams.quantize_channels)
		out_dtype = np.int16

	elif is_mulaw(hparams.input_type):
		#[-1, 1]
		out = mulaw(wav, hparams.quantize_channels)
		constant_values = mulaw(0., hparams.quantize_channels)
		out_dtype = np.float32

	else:
		#[-1, 1]
		out = wav
		constant_values = 0.
		out_dtype = np.float32

	# Compute the mel scale spectrogram from the wav
	#mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
	#mel_frames = mel_spectrogram.shape[1]

	# Compute the mgc,bap,lf0 features

	nFFTHalf, alpha, bap_dim = audio.get_config(hparams.sample_rate)

	mcsize = hparams.num_mgc - 1

	filename = basename #os.path.basename(wav_path).split(".")[0]

	# extract f0,sp,ap
	os.system("analysis %s %s/%s.f0 %s/%s.sp %s/%s.bapd" %
				  (wav_path, lf0_dir, filename,
				   mgc_dir, filename, bap_dir, filename))

	# convert f0 to lf0
	os.system("x2x +da %s/%s.f0 > %s/%s.f0a" %
			  (lf0_dir, filename, lf0_dir, filename))
	os.system("x2x +af %s/%s.f0a | sopr -magic 0.0 -LN "
			  "-MAGIC -1.0E+10 > %s/%s.lf0" %
			  (lf0_dir, filename, lf0_dir, filename))

	# convert　sp to mgc
	os.system("x2x +df %s/%s.sp | sopr -R -m 32768.0 | "
			  "mcep -a %f -m %d -l %d -e 1.0E-8 -j 0 -f 0.0 -q 3 "
			  "> %s/%s.mgc" % (mgc_dir, filename, alpha, mcsize, nFFTHalf, mgc_dir, filename))

	# convert ap to bap
	os.system("x2x +df %s/%s.bapd > %s/%s.bap" %
			  (bap_dir, filename, bap_dir, filename))

	# merge mgc,lf0 and bap to cmp
	os.system("merge +d -s 0 -l 1 -L %d %s/%s.mgc < %s/%s.lf0 > %s/%s.ml" %
			((mcsize+1), mgc_dir, filename, lf0_dir, filename, cmp_dir, filename))
	os.system("merge +d -s 0 -l %d -L %d %s/%s.ml < %s/%s.bap > %s/%s.cmp" %
			((mcsize+2), bap_dim, cmp_dir, filename, bap_dir, filename, cmp_dir, filename))

	#if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
	#	return None

	#Compute the linear scale spectrogram from the wav
	linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	#assert linear_frames == mel_frames

	if hparams.use_lws:
		#Ensure time resolution adjustement between audio and mel-spectrogram
		fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
		l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

		#Zero pad audio signal
		out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
	else:
		#Ensure time resolution adjustement between audio and mel-spectrogram
		l_pad, r_pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams), hparams.wavenet_pad_sides)

		#Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
		out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

	lf0 = np.fromfile("%s/%s.lf0" % (lf0_dir, filename), dtype=np.float32)
	mgc = np.fromfile("%s/%s.mgc" % (mgc_dir, filename), dtype=np.float32)
	bap = np.fromfile("%s/%s.bap" % (bap_dir, filename), dtype=np.float32)
	cmp = np.fromfile("%s/%s.cmp" % (cmp_dir, filename), dtype=np.float32)

	cmp_dim = mcsize + 1 + 1 + bap_dim
	cmp_frames = cmp.shape[0] / cmp_dim
	assert (mgc.shape[0]/(mcsize+1)) == (lf0.shape[0]/1) == (bap.shape[0]/bap_dim) == cmp_frames
	assert cmp_dim == hparams.num_mels
	#assert len(out) >= cmp_frames * audio.get_hop_size(hparams)

	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	#out = out[:mel_frames * audio.get_hop_size(hparams)]
	#assert len(out) % audio.get_hop_size(hparams) == 0
	#time_steps = len(out)

	# Write the spectrogram and audio to disk
	#audio_filename = 'audio-{}.npy'.format(index)
	cmp_mat = cmp.reshape(-1, cmp_dim)
	cmp_filename = 'cmp-{}.npy'.format(basename)
	linear_filename = 'linear-{}.npy'.format(basename)
	#np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	np.save(os.path.join(cmp_dir, cmp_filename), cmp_mat, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (cmp_filename, linear_filename, cmp_frames, text)
