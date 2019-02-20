# -*- coding:utf-8 -*- 
#@Time: 19-1-10 上午8:13
#@Author: Coast Cao
from argparse import ArgumentParser as ARP
import os
import shutil
import glob
import random

def get_config(sr):
	if sr == 16000:
		nFFTHalf = 1024
		alpha = 0.58
		bap_dim = 1

	elif sr == 22050:
		nFFTHalf = 1024
		alpha = 0.65
		bap_dim = 2

	elif sr == 44100:
		nFFTHalf = 2048
		alpha = 0.76
		bap_dim = 5

	elif sr == 48000:
		nFFTHalf = 2048
		alpha = 0.77
		bap_dim = 5
	else:
		raise("ERROR: currently upsupported sampling rate:%d".format(sr))
	return nFFTHalf, alpha, bap_dim

def reconstruct(args):
		cep_dir = os.path.join(args.data_root, "training_data/cmp")
		lpc_dir = os.path.join(args.data_root, "cosy_syn/lpc")
		vocoder_input_dir = os.path.join(args.data_root, "cosy_syn/re_feat")
		resyn_wav_dir = os.path.join(args.data_root, "cosy_syn/wav")
		for d in [lpc_dir, resyn_wav_dir, vocoder_input_dir]:
			if not os.path.exists(d):
				os.makedirs(d)

		cep_files = os.listdir(cep_dir)
		for cep_file in cep_files:
			filename = cep_file.split(".")[0]
			lpc_file = os.path.join(lpc_dir, "%s.lpc" % filename)
			vocoder_input_feat = os.path.join(vocoder_input_dir, "%s.feat" % filename)
			resyn_sw_file = os.path.join(resyn_wav_file, "%s.resyn.sw" % filename)
			resyn_wav_file = os.path.join(resyn_wav_dir, "%s.resyn.wav" % filename)
			os.system("lpc_from_cep %s %s" % (os.path.join(cep_dir, cep_file), lpc_file))
			os.system("paste -d ' ' %s %s > %s" % (os.path.join(cep_dir, cep_file), lpc_file, vocoder_input_feat))
			os.system("test_lpcnet_t %s %s" % (vocoder_input_feat, resyn_sw_file))
			os.system("sox -t sw -r 16000 -c 1 %s -t wav %s" % (resyn_sw_file, resyn_wav_file))


def main():
    arp = ARP(description="extract audio acoustic feature")
    arp.add_argument("--dr", dest="data_root", default="data/tmp",
                     help="root directory of the data")
    arp.add_argument("--sr", dest="sampling_rate", default=22050)
    args = arp.parse_args()

    reconstruct(args)


if __name__ == "__main__":
    main()