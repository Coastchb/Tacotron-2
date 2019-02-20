import os
from hparams import hparams

def read_cep_pitch(file_path):
	cep_pitch = []
	with open(file_path) as fd:
		for line in fd.readlines():
			line = line.strip()
			if(line == "" or line == " "):
				break
			feats = [float(x) for x in line.split()]
			cep_pitch.append(feats[:hparams.num_mels])
	return cep_pitch

