# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
import wave
import os


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, static=False, path=None):
        """
        音频波形动态显示，实时显示波形，实时进行离散傅里叶变换分析频域
        :param static: 是否为静态模式
        :param path:   wav 文件路径
        """
        self.static = static
        if static and os.path.isfile(path):
            self.stream = wave.open(path)
            self.rate = self.stream.getparams()[2]
            self.chunk = self.rate / 2
            self.read = self.stream.readframes
        else:
            self.rate = 2 ** 14
            self.chunk = 2 ** 12
            p = pyaudio.PyAudio()
            self.stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate,
                            input=True, frames_per_buffer=self.chunk)
            self.read = self.stream.read

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        self.t = np.linspace(0, self.chunk - 1, self.chunk)

        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        self.line1, = ax1.plot([], [], lw=2)
        ax1.set_xlim(0, self.chunk)
        ax1.set_ylim(-15000, 15000)

        ax2.set_xlabel('hz')
        ax2.set_ylabel('y')
        self.line2, = ax2.plot([], [], lw=2)
        ax2.set_xlim(0, self.chunk)
        ax2.set_ylim(-50, 100)

        # 更新间隔/ms
        interval = int(1000*self.chunk/self.rate)
        animation.TimedAnimation.__init__(self, fig, interval=interval, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        x = np.linspace(0, self.chunk - 1, self.chunk)
        if self.static:
            # 读取静态wav文件波形
            y = np.fromstring(self.read(self.chunk/2 +1), dtype=np.int16)[:-1]
        else:
            # 实时读取声频
            y = np.fromstring(self.read(self.chunk), dtype=np.int16)

        # 画波形图
        self.line1.set_data(x, y)

        # 傅里叶变化
        freqs = np.linspace(0, self.rate / 2, self.chunk / 2 + 1)
        xf = np.fft.rfft(y) / self.chunk
        xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
        self.line2.set_data(freqs, xfp)

        self._drawn_artists = [self.line1, self.line2]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = [self.line1, self.line2]
        for l in lines:
            l.set_data([], [])


ani = SubplotAnimation()
plt.show()
