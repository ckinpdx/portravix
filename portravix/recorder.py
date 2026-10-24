"""Records exactly what the player was given (frames + 48 kHz audio) to an MP4 for review."""
import os
import subprocess

import imageio
import imageio_ffmpeg
import numpy as np
import scipy.io.wavfile

from .speech_buffer import PLAYBACK_RATE


class Recorder:
    def __init__(self, path, fps=24):
        self.path = path
        self._video_path = path.replace(".mp4", "_video.mp4")
        self._writer = imageio.get_writer(self._video_path, fps=fps, codec="h264", quality=8, macro_block_size=1)
        self._audio = []

    def add(self, frames, audio, speech=False):
        for frame in frames:
            self._writer.append_data(frame)
        self._audio.append(np.asarray(audio, np.float32))

    def close(self):
        self._writer.close()
        wav = self.path.replace(".mp4", ".wav")
        scipy.io.wavfile.write(wav, PLAYBACK_RATE, np.concatenate(self._audio) if self._audio else np.zeros(0, np.float32))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", self._video_path, "-i", wav,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", self.path], check=True)
        os.remove(self._video_path)
        os.remove(wav)
        return self.path
