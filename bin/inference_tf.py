# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "adtof",
#     "pedalboard",
#     "essentia",
#     "librosa",
#     "numba",
#     "pyloudnorm",
#     "numpy<2.0.0",
#     "tf-keras",
# ]
#
# [tool.uv.sources]
# adtof = { path = "/home/xavriley/Projects/ISMIR_LBD_DRUMS_2024/ADTOF" }
# ///
from adtof.model.model import Model
import os
import tensorflow as tf
import pyloudnorm as pyln
import essentia.standard as es
import soundfile as sf
import librosa
import tempfile
import glob
from pedalboard import HighShelfFilter
from pathlib import Path

def __main__():
    import argparse
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "True"
    model, hparams = Model.modelFactory(modelName="Frame_RNN", 
                                        scenario="adtofAll",
                                        fold=0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--hpss", type=bool, default=False)
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    normalize = args.normalize
    hpss = args.hpss

    # create tmp folder
    with tempfile.TemporaryDirectory() as tmpdir:
        # for path in glob.glob(os.path.join(input_folder, "*.wav")):
        for path in glob.glob(input_folder):
            if not path.endswith(".wav") and not path.endswith(".flac"):
                continue

            mix_path = os.path.join(tmpdir, os.path.basename(path))
            midi_out_path = os.path.join(output_folder, os.path.basename(path.replace(".flac", ".wav")) + ".mid")

            if Path(midi_out_path).exists():
                continue
            
            sr = 44100
            if normalize:               
                # this normalizes using the replayGain algorithm to +9db
                eqloader = es.EasyLoader(filename=path, replayGain=9)
                mix = eqloader()
            else:
                mix, sr = sf.read(path)

            if hpss:
                # this normalizes using the replayGain algorithm to 0db
                eqloader = es.EasyLoader(filename=path, replayGain=0)
                mix = eqloader()

                _, y_percussive = librosa.effects.hpss(mix, margin=[1, 5])
                mix = y_percussive

                meter = pyln.Meter(sr) # create BS.1770 meter

                mix = mix.T # when loading via librosa

                loudness = meter.integrated_loudness(mix)
                mix = pyln.normalize.loudness(mix, loudness, -12.0)

            # write mix to tmpdir
            mix_out_path = os.path.join(tmpdir, os.path.basename(path.replace(".flac", ".wav")))
            print(f"writing {mix_out_path}")
            sf.write(mix_out_path, mix, sr)

        result = model.predictFolder(os.path.join(tmpdir, "*.wav"), output_folder, writeMidi=True, **hparams)

if __name__ == "__main__":
    __main__()
else:
    __main__()
