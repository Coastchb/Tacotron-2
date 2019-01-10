# -*- coding:utf-8 -*- 
#@Time: 19-1-10 上午8:13
#@Author: Coast Cao
from argparse import ArgumentParser as ARP
import os
import shutil


def reconstruct(args):
    syn_wav = os.path.join(args.data_root, "syn/wav")
    lf0_dir = os.path.join(args.data_root, "syn/lfo")
    mgc_dir = os.path.join(args.data_root, "syn/mgc")
    bap_dir = os.path.join(args.data_root, "syn/bap")

    # for 16kHz wav
    nFFTHalf = 1024
    alpha = 0.58

    mcsize = 59

    for dire in [lf0_dir, mgc_dir, bap_dir, syn_wav]:
        if (os.path.exists(dire)):
            shutil.rmtree(dire)
        os.makedirs(dire, exist_ok=False)

    lf0_files = os.listdir(lf0_dir)

    for lf0_file in lf0_files:
        filename = lf0_file.split(".")[0]

        # convert lf0 to f0
        os.system("%s/SPTK/sopr -magic -1.0E+10 -EXP -MAGIC 0.0 %s/%s.lf0 | %s/SPTK/x2x +fa > %s/%s.resyn.f0a" %
                  (args.binary_root, lf0_dir, filename, args.binary_root, lf0_dir, filename))
        os.system("%s/SPTK/x2x +ad %s/%s.resyn.f0a > %s/%s.resyn.f0" % (args.binary_root, lf0_dir,
                                                   filename, lf0_dir, filename))
        # convert　mgc to sp
        os.system("%s/SPTK/mgc2sp -a %f -g 0 -m %d -l %d -o 2 %s/%s.mgc | %s/SPTK/sopr -d 32768.0 -P | "
                  "%s/SPTK/x2x +fd > %s/%s.resyn.sp" % (args.binary_root, alpha, mcsize, nFFTHalf,
                                                        mgc_dir, filename, args.binary_root,
                                                        args.binary_root, mgc_dir, filename))
        # convert bap to ap
        os.system("%s/SPTK/x2x +fd %s/%s.bap > %s/%s.resyn.bapd" % (args.binary_root, bap_dir,
                                                              filename, bap_dir, filename))
        # reconstruct wav
        os.system("%s/synth %d %d %s/%s.resyn.f0 %s/%s.resyn.sp %s/%s.resyn.bapd %s/%s.resyn.wav" %
                  (args.binary_root, nFFTHalf, 16000, lf0_dir, filename, mgc_dir, filename, bap_dir,
                  filename, syn_wav, filename))

def main():
    arp = ARP(description="extract audio acoustic feature")
    arp.add_argument("--br", dest="binary_root", default="bin/",
                     help="location of binaries")
    arp.add_argument("--dr", dest="data_root", default="data/LJ",
                     help="root directory of the data")
    args = arp.parse_args()

    reconstruct(args)


if __name__ == "__main__":
    main()