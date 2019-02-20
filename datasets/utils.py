import os

def read_cep_pitch(file_path):
    cep_pitch = []
    for line in os.open(file_path).readlines():
        line = line.strip()
        if(line == "" or line == " "):
            break
        feats = [float(x) for x in line.split()]
        cep_pitch.append(feats)
    return cep_pitch

