# Extrator de caracteristicas
#
# Se lee el contenido de un directorio y, por cada archivo mp3 detectado, se lee el archivo y se extrae las
# características relevantes utilizando la libreria librosa para el procesamiento de señales de audio.

# ==============================================================================
import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from pydub import AudioSegment
import os
import shutil


# ==============================================================================
def extract_feature(path):
    id_song = 1
    feature_set = pd.DataFrame()

    songname_vector = pd.Series(dtype='float64')
    classname_vector = pd.Series()
    label_vector = pd.Series()
    tempo_vector = pd.Series()
    total_beats = pd.Series()
    average_beats = pd.Series()
    chroma_stft_mean = pd.Series()
    chroma_stft_std = pd.Series()
    chroma_stft_var = pd.Series()
    chroma_cq_mean = pd.Series()
    chroma_cq_std = pd.Series()
    chroma_cq_var = pd.Series()
    chroma_cens_mean = pd.Series()
    chroma_cens_std = pd.Series()
    chroma_cens_var = pd.Series()
    mel_mean = pd.Series()
    mel_std = pd.Series()
    mel_var = pd.Series()
    mfcc_mean = pd.Series()
    mfcc_std = pd.Series()
    mfcc_var = pd.Series()
    mfcc_delta_mean = pd.Series()
    mfcc_delta_std = pd.Series()
    mfcc_delta_var = pd.Series()
    rmse_mean = pd.Series()
    rmse_std = pd.Series()
    rmse_var = pd.Series()
    cent_mean = pd.Series()
    cent_std = pd.Series()
    cent_var = pd.Series()
    spec_bw_mean = pd.Series()
    spec_bw_std = pd.Series()
    spec_bw_var = pd.Series()
    contrast_mean = pd.Series()
    contrast_std = pd.Series()
    contrast_var = pd.Series()
    rolloff_mean = pd.Series()
    rolloff_std = pd.Series()
    rolloff_var = pd.Series()
    poly_mean = pd.Series()
    poly_std = pd.Series()
    poly_var = pd.Series()
    tonnetz_mean = pd.Series()
    tonnetz_std = pd.Series()
    tonnetz_var = pd.Series()
    zcr_mean = pd.Series()
    zcr_std = pd.Series()
    zcr_var = pd.Series()
    harm_mean = pd.Series()
    harm_std = pd.Series()
    harm_var = pd.Series()
    perc_mean = pd.Series()
    perc_std = pd.Series()
    perc_var = pd.Series()
    frame_mean = pd.Series()
    frame_std = pd.Series()
    frame_var = pd.Series()

    file_data = [f for f in listdir(path) if isfile(join(path, f))]
    for line in file_data:
        if (line[-1:] == '\n'):
            line = line[:-1]

        songname = path + line
        y, sr = librosa.load(songname, duration=60)
        S = np.abs(librosa.stft(y))

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        poly_features = librosa.feature.poly_features(S=S, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_delta = librosa.feature.delta(mfcc)

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

        # Transforming Features
        songname_vector._set_value(id_song, line)  # song name
        tempo_vector._set_value(id_song, tempo)  # tempo
        total_beats._set_value(id_song, sum(beats))  # beats
        average_beats._set_value(id_song, np.average(beats))
        chroma_stft_mean._set_value(id_song, np.mean(chroma_stft))  # chroma stft
        chroma_stft_std._set_value(id_song, np.std(chroma_stft))
        chroma_stft_var._set_value(id_song, np.var(chroma_stft))
        chroma_cq_mean._set_value(id_song, np.mean(chroma_cq))  # chroma cq
        chroma_cq_std._set_value(id_song, np.std(chroma_cq))
        chroma_cq_var._set_value(id_song, np.var(chroma_cq))
        chroma_cens_mean._set_value(id_song, np.mean(chroma_cens))  # chroma cens
        chroma_cens_std._set_value(id_song, np.std(chroma_cens))
        chroma_cens_var._set_value(id_song, np.var(chroma_cens))
        mel_mean._set_value(id_song, np.mean(melspectrogram))  # melspectrogram
        mel_std._set_value(id_song, np.std(melspectrogram))
        mel_var._set_value(id_song, np.var(melspectrogram))
        mfcc_mean._set_value(id_song, np.mean(mfcc))  # mfcc
        mfcc_std._set_value(id_song, np.std(mfcc))
        mfcc_var._set_value(id_song, np.var(mfcc))
        mfcc_delta_mean._set_value(id_song, np.mean(mfcc_delta))  # mfcc delta
        mfcc_delta_std._set_value(id_song, np.std(mfcc_delta))
        mfcc_delta_var._set_value(id_song, np.var(mfcc_delta))
        rmse_mean._set_value(id_song, np.mean(rmse))  # rmse
        rmse_std._set_value(id_song, np.std(rmse))
        rmse_var._set_value(id_song, np.var(rmse))
        cent_mean._set_value(id_song, np.mean(cent))  # cent
        cent_std._set_value(id_song, np.std(cent))
        cent_var._set_value(id_song, np.var(cent))
        spec_bw_mean._set_value(id_song, np.mean(spec_bw))  # spectral bandwidth
        spec_bw_std._set_value(id_song, np.std(spec_bw))
        spec_bw_var._set_value(id_song, np.var(spec_bw))
        contrast_mean._set_value(id_song, np.mean(contrast))  # contrast
        contrast_std._set_value(id_song, np.std(contrast))
        contrast_var._set_value(id_song, np.var(contrast))
        rolloff_mean._set_value(id_song, np.mean(rolloff))  # rolloff
        rolloff_std._set_value(id_song, np.std(rolloff))
        rolloff_var._set_value(id_song, np.var(rolloff))
        poly_mean._set_value(id_song, np.mean(poly_features))  # poly features
        poly_std._set_value(id_song, np.std(poly_features))
        poly_var._set_value(id_song, np.var(poly_features))
        tonnetz_mean._set_value(id_song, np.mean(tonnetz))  # tonnetz
        tonnetz_std._set_value(id_song, np.std(tonnetz))
        tonnetz_var._set_value(id_song, np.var(tonnetz))
        zcr_mean._set_value(id_song, np.mean(zcr))  # zero crossing rate
        zcr_std._set_value(id_song, np.std(zcr))
        zcr_var._set_value(id_song, np.var(zcr))
        harm_mean._set_value(id_song, np.mean(harmonic))  # harmonic
        harm_std._set_value(id_song, np.std(harmonic))
        harm_var._set_value(id_song, np.var(harmonic))
        perc_mean._set_value(id_song, np.mean(percussive))  # percussive
        perc_std._set_value(id_song, np.std(percussive))
        perc_var._set_value(id_song, np.var(percussive))
        frame_mean._set_value(id_song, np.mean(frames_to_time))  # frames
        frame_std._set_value(id_song, np.std(frames_to_time))
        frame_var._set_value(id_song, np.var(frames_to_time))

        print(songname)
        id_song = id_song + 1

    feature_set['song_name'] = songname_vector  # song name
    feature_set['tempo'] = tempo_vector  # tempo
    feature_set['total_beats'] = total_beats  # beats
    feature_set['average_beats'] = average_beats
    feature_set['chroma_stft_mean'] = chroma_stft_mean  # chroma stft
    feature_set['chroma_stft_std'] = chroma_stft_std
    feature_set['chroma_stft_var'] = chroma_stft_var
    feature_set['chroma_cq_mean'] = chroma_cq_mean  # chroma cq
    feature_set['chroma_cq_std'] = chroma_cq_std
    feature_set['chroma_cq_var'] = chroma_cq_var
    feature_set['chroma_cens_mean'] = chroma_cens_mean  # chroma cens
    feature_set['chroma_cens_std'] = chroma_cens_std
    feature_set['chroma_cens_var'] = chroma_cens_var
    feature_set['melspectrogram_mean'] = mel_mean  # melspectrogram
    feature_set['melspectrogram_std'] = mel_std
    feature_set['melspectrogram_var'] = mel_var
    feature_set['mfcc_mean'] = mfcc_mean  # mfcc
    feature_set['mfcc_std'] = mfcc_std
    feature_set['mfcc_var'] = mfcc_var
    feature_set['mfcc_delta_mean'] = mfcc_delta_mean  # mfcc delta
    feature_set['mfcc_delta_std'] = mfcc_delta_std
    feature_set['mfcc_delta_var'] = mfcc_delta_var
    feature_set['rmse_mean'] = rmse_mean  # rmse
    feature_set['rmse_std'] = rmse_std
    feature_set['rmse_var'] = rmse_var
    feature_set['cent_mean'] = cent_mean  # cent
    feature_set['cent_std'] = cent_std
    feature_set['cent_var'] = cent_var
    feature_set['spec_bw_mean'] = spec_bw_mean  # spectral bandwidth
    feature_set['spec_bw_std'] = spec_bw_std
    feature_set['spec_bw_var'] = spec_bw_var
    feature_set['contrast_mean'] = contrast_mean  # contrast
    feature_set['contrast_std'] = contrast_std
    feature_set['contrast_var'] = contrast_var
    feature_set['rolloff_mean'] = rolloff_mean  # rolloff
    feature_set['rolloff_std'] = rolloff_std
    feature_set['rolloff_var'] = rolloff_var
    feature_set['poly_mean'] = poly_mean  # poly features
    feature_set['poly_std'] = poly_std
    feature_set['poly_var'] = poly_var
    feature_set['tonnetz_mean'] = tonnetz_mean  # tonnetz
    feature_set['tonnetz_std'] = tonnetz_std
    feature_set['tonnetz_var'] = tonnetz_var
    feature_set['zcr_mean'] = zcr_mean  # zero crossing rate
    feature_set['zcr_std'] = zcr_std
    feature_set['zcr_var'] = zcr_var
    feature_set['harm_mean'] = harm_mean  # harmonic
    feature_set['harm_std'] = harm_std
    feature_set['harm_var'] = harm_var
    feature_set['perc_mean'] = perc_mean  # percussive
    feature_set['perc_std'] = perc_std
    feature_set['perc_var'] = perc_var
    feature_set['frame_mean'] = frame_mean  # frames
    feature_set['frame_std'] = frame_std
    feature_set['frame_var'] = frame_var

    feature_set.to_csv('temp.csv')
    feature_set.to_json('temp.json')
