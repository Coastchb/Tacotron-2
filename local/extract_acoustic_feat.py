# -*- coding:utf-8 -*- 
#@Time: 19-1-9 下午9:51
#@Author: Coast Cao
from argparse import ArgumentParser as ARP
import os
import shutil

def extract_acoustic_feat(args):
    wav_resamp = os.path.join(args.data_root, "wav_re")
    lf0_dir = os.path.join(args.data_root, "lf0")
    mgc_dir = os.path.join(args.data_root, "mgc")
    bap_dir = os.path.join(args.data_root, "bap")

    # for 16kHz wav
    tar_fs = 16000
    nFFTHalf = 1024
    alpha = 0.58
    mcsize = 59

    for dire in [lf0_dir, mgc_dir, bap_dir, wav_resamp]:
        if(os.path.exists(dire)):
            shutil.rmtree(dire)
        os.makedirs(dire, exist_ok=False)

    wav_dir = os.path.join(args.data_root, "wav")
    wav_files = os.listdir(wav_dir)
    for wav_file in wav_files:
        filename = wav_file.split(".")[0]
        # currently just downsample wav to 16kHz
        os.system("sox %s/%s -r %d %s/%s" % (wav_dir, wav_file, tar_fs, wav_resamp, wav_file))
        # extract f0,sp,ap
        os.system("%s/WORLD/analysis %s/%s %s/%s.f0 %s/%s.sp %s/%s.bapd" %
                  (args.binary_root, wav_resamp, wav_file, lf0_dir, filename,
                  mgc_dir, filename, bap_dir, filename))
        # convert f0 to lf0
        os.system("%s/SPTK/x2x +da %s/%s.f0 > %s/%s.f0a" %
                  (args.binary_root, lf0_dir, filename, lf0_dir, filename))
        os.system("%s/SPTK/x2x +af %s/%s.f0a | %s/SPTK/sopr -magic 0.0 -LN "
                  "-MAGIC -1.0E+10 > %s/%s.lf0" % (args.binary_root, lf0_dir,
                                                   filename, args.binary_root,
                                                   lf0_dir, filename))
        # convert　sp to mgc
        os.system("%s/SPTK/x2x +df %s/%s.sp | %s/SPTK/sopr -R -m 32768.0 | "
                  "%s/SPTK/mcep -a %f -m %d -l %d -e 1.0E-8 -j 0 -f 0.0 -q 3 "
                  "> %s/%s.mgc" % (args.binary_root, mgc_dir, filename,args.binary_root,
                                   args.binary_root, alpha, mcsize, nFFTHalf, mgc_dir, filename))
        # convert ap to bap
        os.system("%s/SPTK/x2x +df %s/%s.bapd > %s/%s.bap" % (args.binary_root, bap_dir,
                                                              filename, bap_dir, filename))

def main():
    arp = ARP(description="extract audio acoustic feature")
    arp.add_argument("--br", dest="binary_root", default="bin/",
                     help="location of binaries")
    arp.add_argument("--dr", dest="data_root", default="data/LJ",
                     help="root directory of the data")
    args = arp.parse_args()

    extract_acoustic_feat(args)


if __name__ == "__main__":
    main()