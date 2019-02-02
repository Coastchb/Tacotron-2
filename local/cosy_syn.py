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

    lf0_dir = os.path.join(args.data_root, "lf0")
    mgc_dir = os.path.join(args.data_root, "mgc")
    bap_dir = os.path.join(args.data_root, "bap")

    syn_wav = os.path.join(args.data_root, "syn/wav")
    syn_f0 = os.path.join(args.data_root, "syn/f0")
    syn_sp = os.path.join(args.data_root, "syn/sp")
    syn_ap = os.path.join(args.data_root, "syn/ap")

    # for 16kHz wav
    nFFTHalf, alpha, _ = get_config(args.sampling_rate)

    mcsize = 59

    for dire in [syn_wav, syn_f0, syn_sp, syn_ap]:
        if (os.path.exists(dire)):
            shutil.rmtree(dire)
        os.makedirs(dire, exist_ok=False)

    lf0_files = glob.glob(lf0_dir + "/*.lf0")
    random.shuffle(lf0_files)
    lf0_files = lf0_files[:10]

    print("To construct %d wavs" % len(lf0_files))

    for lf0_file in lf0_files:
        filename = lf0_file.split("/")[-1].split(".")[0]

        os.system("%s/SPTK/sopr -magic -1.0E+10 -EXP -MAGIC 0.0 %s/%s.lf0 | %s/SPTK/x2x +fa > %s/%s.resyn.f0a" %
                  (args.binary_root, lf0_dir, filename, args.binary_root, syn_f0, filename))
        os.system("%s/SPTK/x2x +ad %s/%s.resyn.f0a > %s/%s.resyn.f0" % (args.binary_root, syn_f0,
                                                   filename, syn_f0, filename))
        # convert　mgc to sp
        os.system("%s/SPTK/mgc2sp -a %f -g 0 -m %d -l %d -o 2 %s/%s.mgc | %s/SPTK/sopr -d 32768.0 -P | "
                  "%s/SPTK/x2x +fd > %s/%s.resyn.sp" % (args.binary_root, alpha, mcsize, nFFTHalf,
                                                        mgc_dir, filename, args.binary_root,
                                                        args.binary_root, syn_sp, filename))
        # convert bap to ap
        os.system("%s/SPTK/x2x +fd %s/%s.bap > %s/%s.resyn.bapd" % (args.binary_root, bap_dir,
                                                              filename, syn_ap, filename))
        # reconstruct wav
        os.system("%s/WORLD/synth %d %d %s/%s.resyn.f0 %s/%s.resyn.sp %s/%s.resyn.bapd %s/%s.resyn.wav" %
                  (args.binary_root, nFFTHalf, args.sampling_rate, syn_f0, filename, syn_sp, filename, syn_ap,
                  filename, syn_wav, filename))

def main():
    arp = ARP(description="extract audio acoustic feature")
    arp.add_argument("--br", dest="binary_root", default="bin/",
                     help="location of binaries")
    arp.add_argument("--dr", dest="data_root", default="data/tmp",
                     help="root directory of the data")
    arp.add_argument("--sr", dest="sampling_rate", default=22050)
    args = arp.parse_args()

    reconstruct(args)


if __name__ == "__main__":
    main()