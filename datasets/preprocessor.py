import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from datasets import audio
from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize
import shutil
from nnmnkwii.preprocessing import interp1d
import soundfile as sf

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
		#if (os.path.exists(dire)):
		#	shutil.rmtree(dire)
		os.makedirs(dire, exist_ok=True)
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

	if hparams.trim_silence:
		tar_wavfile = wav_path[:-4] + "_trim.wav"
		print("raw wav path:%s" % wav_path)
		wav_raw, fs = sf.read(wav_path)
		wav_trim = audio.trim_silence(wav_raw, hparams)
		sf.write(tar_wavfile, wav_trim, fs)

		wav_path = tar_wavfile

	nFFTHalf, alpha, bap_dim = audio.get_config(hparams.sample_rate)

	mcsize = hparams.num_mgc - 1

	filename = basename #os.path.basename(wav_path).split(".")[0]

	print('extract feats for %s' % wav_path)

	# extract f0,sp,ap
	os.system("analysis %s %s/%s.f0 %s/%s.sp %s/%s.bapd" %
				  (wav_path, lf0_dir, filename,
				   mgc_dir, filename, bap_dir, filename)) # get float64???

    # interpolate f0
	f0 = np.fromfile("%s/%s.f0" % (lf0_dir, filename),dtype=np.float64)
	continuous_f0 = interp1d(f0, kind="slinear")
	continuous_f0.tofile("%s/%s.f0c" % (lf0_dir, filename))

	# convert f0 to lf0
	os.system("x2x +da %s/%s.f0c > %s/%s.f0a" % (lf0_dir, filename, lf0_dir, filename))
	os.system("x2x +af %s/%s.f0a | sopr -magic 0.0 -LN -MAGIC -1.0E+10 > %s/%s.lf0" % (
		lf0_dir, filename, lf0_dir, filename))

	# convert　sp to mgc
	os.system("x2x +df %s/%s.sp | sopr -R -m 32768.0 | "
			  "mcep -a %f -m %d -l %d -e 1.0E-8 -j 0 -f 0.0 -q 3 "
			  "> %s/%s.mgc" % (mgc_dir, filename, alpha, mcsize, nFFTHalf, mgc_dir, filename))

	# convert ap to bap
	os.system("x2x +df %s/%s.bapd > %s/%s.bap" %
			  (bap_dir, filename, bap_dir, filename))

	# merge mgc,lf0 and bap to cmp
	os.system("merge +f -s 0 -l 1 -L %d %s/%s.mgc < %s/%s.lf0 > %s/%s.ml" %
			((mcsize+1), mgc_dir, filename, lf0_dir, filename, cmp_dir, filename))
	os.system("merge +f -s 0 -l %d -L %d %s/%s.ml < %s/%s.bap > %s/%s.cmp" %
			(bap_dim, (mcsize+2), cmp_dir, filename, bap_dir, filename, cmp_dir, filename))

	#if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
	#	return None

	#Compute the linear scale spectrogram from the wav
	wav = audio.load_wav(wav_path, hparams.sample_rate)
	linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	#assert linear_frames == mel_frames

	lf0 = np.fromfile("%s/%s.lf0" % (lf0_dir, filename), dtype=np.float32)
	mgc = np.fromfile("%s/%s.mgc" % (mgc_dir, filename), dtype=np.float32)
	bap = np.fromfile("%s/%s.bap" % (bap_dir, filename), dtype=np.float32)
	cmp = np.fromfile("%s/%s.cmp" % (cmp_dir, filename), dtype=np.float32)

	cmp_dim = mcsize + 1 + 1 + bap_dim
	cmp_frames = cmp.shape[0] / cmp_dim
	#print(f0[:100])
	#print(continuous_f0[:100])
	print(lf0.shape)
	print(continuous_f0.shape)
	print(mgc.shape)
	print(bap.shape)
	print(cmp_frames)
	print(continuous_f0.dtype)
	print(mgc.dtype)
	print(bap.dtype)
	assert (mgc.shape[0]/(mcsize+1)) == (continuous_f0.shape[0]/1) == (bap.shape[0]/bap_dim) == cmp_frames
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
