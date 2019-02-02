import argparse
from multiprocessing import cpu_count
import os
from tqdm import tqdm
from datasets import preprocessor
from hparams import hparams
import glob
import numpy as np

def preprocess(args, input_folder, out_dir, hparams):
	cmp_dir = os.path.join(out_dir, 'cmp')
	linear_dir = os.path.join(out_dir, 'linear')
	for d in [cmp_dir, linear_dir]:
		#if(os.path.exists(d)):
		#	shutil.rmtree(d)
		os.makedirs(d, exist_ok=True)

	metadata = preprocessor.build_from_path(hparams, input_folder, cmp_dir, linear_dir, args.n_jobs, tqdm=tqdm)
	write_metadata(metadata, out_dir)

def compute_mean_var(dire):
	files = glob.glob(os.path.join(dire,"cmp/cmp-*.npy"))
	cmps = []
	for file in files:
		cmps.extend(np.load(file))
	mean = np.mean(cmps, axis=0)
	std = np.std(cmps, axis=0)

	np.save(os.path.join(dire,"cmp-mean.npy"), mean, allow_pickle=False)
	np.save(os.path.join(dire,"cmp-var.npy"), std, allow_pickle=False)

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	compute_mean_var(out_dir)
	cmp_frames = sum([int(m[2]) for m in metadata])
	#timesteps = sum([int(m[3]) for m in metadata])
	#sr = hparams.sample_rate
	#hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames'.format(
		len(metadata), cmp_frames))
	#print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max cmp frames length: {}'.format(max(int(m[2]) for m in metadata)))
	#print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

def run_preprocess(args, hparams):
	input_folder = os.path.join(args.base_dir, args.dataset)
	output_folder = os.path.join(input_folder, args.output)

	preprocess(args, input_folder, output_folder, hparams)


def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='data')
	parser.add_argument('--hparams', default='', 
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--dataset', default='LJSpeech-1.1')
	parser.add_argument('--language', default='en_US')
	parser.add_argument('--voice', default='female')
	parser.add_argument('--reader', default='mary_ann')
	parser.add_argument('--merge_books', default='False')
	parser.add_argument('--book', default='northandsouth')
	parser.add_argument('--output', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	modified_hp = hparams.parse(args.hparams)

	assert args.merge_books in ('False', 'True')

	run_preprocess(args, modified_hp)


if __name__ == '__main__':
	main()