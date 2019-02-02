import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import os


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr, inv_preemphasize, k):
    wav = inv_preemphasis(wav, k, inv_preemphasize)
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


# From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def trim_silence(wav, hparams):
    '''Trim leading and trailing silence
    Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
    '''
    # Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
    return librosa.effects.trim(wav, top_db=hparams.trim_top_db, frame_length=hparams.trim_fft_size,
                                hop_length=hparams.trim_hop_size)[0]


def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size


def linearspectrogram(wav, hparams):
    # D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    D = _stft(wav, hparams)
    S = _amp_to_db(np.abs(D) ** hparams.magnitude_power, hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def melspectrogram(wav, hparams):
    # D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D) ** hparams.magnitude_power, hparams), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def inv_linear_spectrogram(linear_spectrogram, hparams):
    '''Converts linear spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + hparams.ref_level_db) ** (1 / hparams.magnitude_power)  # Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)


def inv_mel_spectrogram(mel_spectrogram, hparams):
    '''Converts mel spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db) ** (1 / hparams.magnitude_power),
                       hparams)  # Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)


def _lws_processor(hparams):
    import lws
    return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")


def _griffin_lim(S, hparams):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y


def _stft(y, hparams):
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size,
                            pad_mode='constant')


def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * (
                        (S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0,
                           hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * (
                    (S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))


def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value,
                              hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (
                                 2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((np.clip(D, 0,
                             hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (
                    2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)


def normalize_tf(S, hparams):
    # [0, 1]
    if hparams.normalize_for_wavenet:
        if hparams.allow_clipping_in_normalization:
            return tf.minimum(tf.maximum((S - hparams.min_level_db) / (-hparams.min_level_db),
                                         -hparams.max_abs_value), hparams.max_abs_value)

        else:
            return (S - hparams.min_level_db) / (-hparams.min_level_db)

    # [-max, max] or [0, max]
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return tf.minimum(tf.maximum((2 * hparams.max_abs_value) * (
                        (S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                                         -hparams.max_abs_value), hparams.max_abs_value)
        else:
            return tf.minimum(
                tf.maximum(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0),
                hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * (
                    (S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))


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
        raise ("ERROR: currently upsupported sampling rate:%d".format(sr))
    return nFFTHalf, alpha, bap_dim


def synthesis(cmp_p, syn_dir, filename, hparams):
    nFFTHalf, alpha, bap_dim = get_config(hparams.sample_rate)
    mcsize = hparams.num_mgc - 1

    base_dir = os.path.join(os.path.dirname(os.path.dirname(syn_dir)),"training_data") #"/home/potato/Tacotron-2/data/LJSpeech-1.1/training_data"
    feat_mean = np.load(os.path.join(base_dir, "cmp-mean.npy"))
    feat_var = np.load(os.path.join(base_dir, "cmp-var.npy"))
    print("#### in synthesis###")
    print(cmp_p.shape)
    cmp_p = cmp_p * feat_var + feat_mean
    print(cmp_p.shape)

    mgc = cmp_p[:, 0:hparams.num_mgc]
    lf0 = cmp_p[:, -3]
    bap = cmp_p[:, -2:]

    lf0 = lf0.astype(np.float32)
    mgc = mgc.astype(np.float32)
    bap = bap.astype(np.float32)

    lf0.tofile("%s/%s.lf0" % (syn_dir, filename))
    mgc.tofile("%s/%s.mgc" % (syn_dir, filename))
    bap.tofile("%s/%s.bap" % (syn_dir, filename))

    # convert lf0 back to f0
    os.system("sopr -magic -1.0E+10 -EXP -MAGIC 0.0 %s/%s.lf0 | x2x +fa > %s/%s.resyn.f0a" %
              (syn_dir, filename, syn_dir, filename))
    os.system("x2x +ad %s/%s.resyn.f0a > %s/%s.resyn.f0" % (syn_dir, filename, syn_dir, filename))

    # convert　mgc to sp
    os.system("mgc2sp -a %f -g 0 -m %d -l %d -o 2 %s/%s.mgc | sopr -d 32768.0 -P | "
              "x2x +fd > %s/%s.resyn.sp" % (alpha, mcsize, nFFTHalf,
                                            syn_dir, filename, syn_dir, filename))
    # convert bap to ap
    os.system("x2x +fd %s/%s.bap > %s/%s.resyn.bapd" %
              (syn_dir, filename, syn_dir, filename))

    syn_wav = os.path.join(syn_dir, 'wav')
    os.makedirs(syn_wav, exist_ok=True)
    # reconstruct wav
    os.system("synth %d %d %s/%s.resyn.f0 %s/%s.resyn.sp %s/%s.resyn.bapd %s/%s.resyn.wav" %
              (nFFTHalf, hparams.sample_rate, syn_dir, filename, syn_dir, filename, syn_dir,
               filename, syn_wav, filename))
    return load_wav(os.path.join(syn_wav, "%s.resyn.wav" % filename), hparams.sample_rate)
