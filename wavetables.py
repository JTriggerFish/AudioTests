"""
Copyright (c) 2015 JTriggerFish

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy
import matplotlib.pyplot as plt
import math, wave, array
import multirate

WAVETABLE_SIZE = 2047


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    import numpy as np
    # __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    # __version__ = "1.0.4"
    # __license__ = "MIT"

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    import numpy as np
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best')  # , framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


def sine(frequency):
    t = numpy.arange(0, WAVETABLE_SIZE) / float(WAVETABLE_SIZE - 1)
    if frequency >= WAVETABLE_SIZE / 2:
        return t * 0
    t[-1] = t[0]
    x = numpy.sin(2 * numpy.pi * t * frequency)
    return x


class OscFromPartials():
    def __init__(self, partials=None, amplitudes=None):
        self.partials = partials
        self.amplitudes = amplitudes

    def __call__(self, baseFreq, numHarmonics):
        x = 0
        for (p, ampl) in zip(self.partials, self.amplitudes):
            if p > numHarmonics:
                break
            a = ampl
            x += a * sine(p)
        return x

    def ReconstructFromSignal(self,x):
        import scipy.optimize as optimize

        x  = x / max(abs(x))
        X_ = numpy.fft.rfft(x)
        n  = x.size
        t = numpy.arange(0, n) / float(n - 1)
        t[-1] = t[0]
        targetAs = abs(X_)
        N = n/2
        freqs = numpy.arange(1,N+1)

        initGuess = numpy.zeros(N)
        initGuess[0] = 1.0

        def distance(partialAs):
            y = 0
            for (f,p) in zip(freqs, partialAs**2):
                y += p * numpy.sin(2 * numpy.pi * t * f)
            y = y / max(abs(y))
            return targetAs - abs(numpy.fft.rfft(y))


        self.partials   = freqs
        self.amplitudes, fit = optimize.leastsq(distance, x0 = initGuess)
        self.amplitudes **=2

        plotSignals(x, self(1.0, N))



class OscFromTable():
    def __init__(self, table, sampleRate=None):
        self.table = multirate.resample(table, WAVETABLE_SIZE, table.size)
        if sampleRate is None:
            sampleRate = 44800
        self.sampleRate = sampleRate

    def __call__(self, basefreq, numharmonics):
        x = 0
        cutoff = min(.999, float(numharmonics*2.0)  / self.table.size)
        #TODO FIXME
        if False:#cutoff < 0.999:
            x = lowPassSignal(self.table, cutoff)
        else:
            x = numpy.copy(self.table)
        return x

    def setName(self, name):
        self.name = name


def comb(_, n):
    x = 0
    for i in xrange(n):
        x += sine(i + 1)
    return x


def tri(_, n, f=1):
    x = 0
    for i in xrange(int((n - 0.01) / 2) + 1):
        x += sine((2 * i + 1) * f) / (2 * i + 1) ** 2.0
    return x


def tri_stack_bright(n):
    x = 0
    for i in xrange(n):
        x += tri(15 + 5 * n, i + n / 3)
    return x


def tri_stack(n):
    x = 0
    for i in xrange(n):
        x += tri(5 + 7 * n, i + 1) / ((i + 1) ** 0.5)
    return x


def saw(_, nh, f=1):
    x = 0
    for i in xrange(nh):
        if f * (i + 1) > nh:
            break
        x += sine((i + 1) * f) / (i + 1)
    return x


def saw_stack(_, nh, nstack=7):
    x = 0
    for i in xrange(nstack):
        nsubh = min(1 + 6 * i, nh)
        x += saw(nsubh, i + 1) / ((i + 1) ** 0.5)
    return x


def square(_, n):
    x = 0
    for i in xrange(int((n - 0.01) / 2) + 1):
        x += sine(2 * i + 1) / (2 * i + 1)
    return x


def quadra(n):
    x = 0
    for harmonic, amplitude in zip(xrange(4), [1, 0.5, 1, 0.5]):
        x += sine(2 * n + 2 * harmonic + 1) * amplitude
    return x


def drawbars(bars):
    pipes = [1.0, 3.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    x = 0
    for intensity, frequency in zip(bars, pipes):
        x += int(intensity) / 8.0 * sine(frequency)
    return x


def pulse(duty):
    t = numpy.arange(0, WAVETABLE_SIZE) / float(WAVETABLE_SIZE - 1)
    t[-1] = t[0]
    t[t < duty] = -1.0
    t[t >= duty] = 1.0
    return -t


def burst(duty):
    t = numpy.arange(0, WAVETABLE_SIZE) / float(WAVETABLE_SIZE - 1)
    t[-1] = t[0]
    d = duty ** 0.5
    t[t < d] = -1.0
    t[t >= d] = 0.0
    return -t * sine(1.0 / duty)


def hybrid(duty):
    cycle = (numpy.arange(0, WAVETABLE_SIZE) + int((duty - 0.5) * WAVETABLE_SIZE)) % WAVETABLE_SIZE
    x = pulse(duty)
    x += saw(80)[cycle]
    x -= (x.mean())
    return x


def trisaw(harmonic):
    return tri(80) + saw(80, harmonic) * (1 if harmonic != 1 else 0) * 0.5


def square_formant(nh, ratio=5.0):
    t = numpy.arange(0, WAVETABLE_SIZE) / float(WAVETABLE_SIZE - 1)
    phase = t * (ratio ** 0.5) * 0.5
    phase[phase >= 1.0] = 1.0
    amplitude = numpy.cos(phase * numpy.pi) + 1
    formant = (sine(ratio * 0.75) + 1.0) * amplitude * 0.5
    formant -= (formant.max() + formant.min()) / 2.0
    return formant


def saw_formant(ratio):
    t = numpy.arange(0, WAVETABLE_SIZE) / float(WAVETABLE_SIZE - 1)
    amplitude = 1.0 - t
    formant = (sine(ratio) + 1.0) * amplitude * 0.5
    formant -= (formant.max() + formant.min()) / 2.0
    return formant


def bandpass_formant(ratio):
    t = numpy.arange(0, WAVETABLE_SIZE) / float(WAVETABLE_SIZE - 1)
    amplitude = 1.0 - t
    formant = sine(ratio * 1.5) * amplitude * 0.5
    return formant


def sine_power(power):
    x = sine(1.0)
    if power >= 6:
        x = sine(2.0) * 2.0
    x += saw(16)
    power = 2.0 ** (1.2 * (power - 0.5))
    return numpy.sign(x) * (numpy.abs(x) ** power)


def formant_f(index):
    formant_1 = 3.9 * (index + 1) / 8.0
    formant_2 = (1.0 - numpy.cos(formant_1 * numpy.pi * 0.8))
    t = numpy.arange(0, WAVETABLE_SIZE) / float(WAVETABLE_SIZE - 1)
    amplitude_1 = (1.0 - t) ** 0.2 * numpy.exp(-4.0 * t)
    amplitude_2 = (1.0 - t) ** 0.2 * numpy.exp(-2.0 * t)
    formant_3 = sine(1 + 2.8 * (formant_2 + formant_1)) * amplitude_2 * 1.7
    formant_1 = sine(1 + 3 * formant_1) * amplitude_1
    formant_2 = sine(1 + 4 * formant_2) * amplitude_2 * 1.5
    f = formant_1 + formant_2 + formant_3
    return f - (f.max() + f.min()) / 2.0


def distort(x):
    return numpy.arctan(x * 6.0) / numpy.pi


def digi_formant_f(index):
    formant_1 = 3.8 * (index + 1) / 8.0
    formant_2 = (1.0 - numpy.cos(formant_1 * numpy.pi * 0.4))
    t = numpy.arange(0, WAVETABLE_SIZE) / float(WAVETABLE_SIZE - 1)
    amplitude_1 = (1.0 - t) ** 0.2 * numpy.exp(-4.0 * t)
    amplitude_2 = (1.0 - t) ** 0.2 * numpy.exp(-2.0 * t)
    formant_3 = distort(sine(1 + 2.9 * (formant_2 + formant_1))) * amplitude_2 * 0.7
    formant_1 = distort(sine(1 + 3.2 * formant_1)) * amplitude_1
    formant_2 = distort(sine(1 + 4.1 * formant_2)) * amplitude_2 * 0.7
    f = formant_1 + formant_2 + formant_3
    return f - (f.max() + f.min()) / 2.0


def midiToFreq(midiNote):
    return 440.0 * pow(2.0, (midiNote - 69) / 12)


def makeMipMap(waveFn, splitsPerOctave=None, targetSampleRate=None):
    if splitsPerOctave is None:
        splitsPerOctave = 3
    if targetSampleRate is None:
        targetSampleRate = 44800
    # In Hz
    lowestFreq = 20.
    # highestFreq = 7040

    numOctaves = 10
    freqSplits = [lowestFreq * 2 ** (i / float(splitsPerOctave)) for i in xrange(0, splitsPerOctave * numOctaves)]
    numHarmonics = [int(targetSampleRate / 2. / f) for f in freqSplits]
    tables = map(waveFn, freqSplits, numHarmonics)

    # renormalise
    tables = [t / max(abs(t)) for t in tables]

    """
    plt.figure()
    #plt.plot(tables[19])
    plt.plot(tables[20])
    #plt.plot(tables[21])
    plt.show()
    """

    return (freqSplits, tables)


def interpWave(table, phase):
    """phase is between 0 and 1"""
    L = table.size - 1  # The last sample repeats the first one
    t = phase * L
    inext = int(t) + 1
    i = inext - 1
    return table[i] + (table[inext] - table[i]) * (t - i)


def waveTableOscillator(samples, waveTable, freqFn, sampleRate=None):
    if sampleRate is None:
        sampleRate = 44800

    freqSplits, waves = waveTable

    phase = 0.
    freq = freqFn(samples)
    output = []
    for i in samples:
        f = freq[i]

        tableIdx = numpy.searchsorted(freqSplits, f) - 1
        tableIdx = min(tableIdx, len(freqSplits) - 1)
        tableNext = min(tableIdx + 1, len(freqSplits) - 1)

        freq1 = freqSplits[tableIdx]
        freq2 = freqSplits[tableNext]

        phase += f / sampleRate
        phase %= 1.0

        wave1 = interpWave(waves[tableIdx], phase)
        wave2 = interpWave(waves[tableNext], phase)
        v = wave1 + (f - freq1) / (freq2 - freq1) * (wave2 - wave1)

        output.append(v)

    return numpy.array(output)


def freqRamp(samples, freqStart, freqStop):
    return numpy.exp(numpy.log(freqStart) + samples * numpy.log(freqStop / freqStart) / (samples.size - 1))


def outputFiles(names, soundStreams, sampleRate=None):
    if sampleRate is None:
        sampleRate = 48000
    assert (len(names) == len(soundStreams))
    volume = 1
    numChan = 1  # of channels (1: mono, 2: stereo)
    dataSize = 2  # 2 bytes because of using signed short integers => bit depth = 16

    for name, sound in zip(names, soundStreams):
        data = array.array('h')  # signed short integer (-32768 to 32767) data
        N = sound.size

        for i in range(N):
            sample = 32767 * float(volume)
            sample *= sound[i]
            data.append(int(sample))

        f = wave.open(name + '.wav', 'w')
        f.setparams((numChan, dataSize, sampleRate, N, "NONE", "Uncompressed"))
        f.writeframes(data.tostring())
        f.close()


class WavFile:
    def __init__(self, fileName):
        import struct
        f = wave.open(fileName, 'r')
        self.nChannels = f.getnchannels()
        self.depth = f.getsampwidth()  # in bytes
        self.sampleRate = f.getframerate()
        self.size = f.getnframes()

        frames = f.readframes(self.size * self.nChannels)
        out = struct.unpack_from("%dh" % self.size * self.nChannels, frames)

        if self.nChannels == 2:
            self.left = numpy.array(out[0::2], dytpe='float32')
            self.right = numpy.array(out[1::2], dtype='float32')
            self.left /= 2 ** (self.depth * 8 - 1)
            self.right /= 2 ** (self.depth * 8 - 1)


        else:
            self.left = numpy.array(out, dtype='float32')
            self.left /= 2 ** (self.depth * 8 - 1)
            self.right = self.left

        f.close()


def freqSweep(wavetables):
    duration = 10  # seconds
    freqStart = 20
    freqStop = 12000
    sampleRate = 48000
    numSamples = duration * sampleRate
    samples = numpy.arange(numSamples)

    tables = [makeMipMap(w) for w in wavetables]

    names = ["freqSweep_" + w.__name__ for w in wavetables]
    soundStreams = []

    for wt in tables:
        soundStreams.append(
            waveTableOscillator(samples, wt, lambda s: freqRamp(s, freqStart, freqStop), sampleRate=sampleRate))

    outputFiles(names, soundStreams, sampleRate)


def freqHarmonics(wavetables):
    import inspect

    durationPerNote = 2.  # seconds
    freqStart = 100.
    sampleRate = 48000
    harmonics = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100]
    # harmonics = [1]
    numSamples = int(durationPerNote * len(harmonics) * sampleRate)
    samples = numpy.arange(numSamples)

    tables = [makeMipMap(w) for w in wavetables]

    names = []
    for w in wavetables:
        if inspect.isfunction(w):
            names.append(w.__name__)
        else:
            names.append(w.name)
    names = ["freqHarmonics_" + n for n in names]
    soundStreams = []

    def freqFn():
        freqFn.durationPerNote = durationPerNote
        freqFn.sampleRate = sampleRate
        freqFn.harmonics = harmonics
        freqFn.freqStart = freqStart

        def f(samples):
            i = 0
            idx = 0
            freq = []
            for s in samples:
                if i > freqFn.sampleRate * freqFn.durationPerNote:
                    idx += 1
                    i = 0
                idx = min(idx, len(freqFn.harmonics) - 1)
                i += 1
                freq.append(freqFn.freqStart * freqFn.harmonics[idx])
            return freq

        def f2(samples):
            return numpy.array([100.0] * samples.size)

        return f

    for wt in tables:
        soundStreams.append(waveTableOscillator(samples, wt, freqFn(), sampleRate=sampleRate))

    outputFiles(names, soundStreams, sampleRate)


def findPartialsFromPeaks(soundArray, sampleRate, show=False):
    fft = numpy.fft.rfft(numpy.blackman(soundArray.size) * soundArray)

    absVals = abs(2 * fft / fft.size)
    dbVals = 20 * numpy.log10(absVals)
    peaks = detect_peaks(dbVals, show=True)
    peaks = [p for p in peaks if dbVals[p] > -60.]  # 60db threshold for now

    # print peaks
    freqs = numpy.array(peaks) * float(sampleRate) / soundArray.size
    partials = freqs / freqs[0]
    amplitudes = numpy.array([absVals[p] / absVals[peaks[0]] for p in peaks])

    print partials, amplitudes


def sawFromPartials():
    partials = numpy.array(xrange(1, 10000))
    amplitudes = 1. / partials
    return OscFromPartials(partials, amplitudes)



def lowPassSignal(x, cutoff):
    """Steep FIR low pass at cutoff, using window method"""
    """Cutoff is between 0 and 1.0, 1.0 being Nyquist"""
    import scipy.signal as signal
    order = 80
    b = signal.firwin(order, cutoff=cutoff, window="hamming")
    a = 1
    y = signal.lfilter(b, a, x)
    return y


def plotSignals(*args):
    # File
    plt.figure(1)
    plt.subplot(211)
    for x in args:
        t = numpy.arange(0, x.size) / float(x.size - 1)
        plt.plot(t, x)

    # FFT
    plt.subplot(212)
    for x in args:
        #fft = numpy.fft.rfft(numpy.blackman(x.size) * x)
        fft = numpy.fft.rfft(numpy.blackman(x.size) * x)
        absVals = abs(2 * fft / fft.size)
        dbVals = 20 * numpy.log10(absVals)
        f = numpy.arange(0, fft.size) / float(fft.size - 1) * math.pi
        plt.plot(f, dbVals)

    plt.show()


def plotWav(fileName):
    wav = WavFile(fileName)
    plotSignal(wav.left)

    # wav.left = lowPassSignal(wav.left, 0.5)


def waveFoldSignal(strength, SR=None, f=None):
    if SR is None:
        SR = 44800
    if f is None:
        f = 1.0
    t = numpy.arange(0, SR) / float(SR - 1)
    x = numpy.sin(2 * numpy.pi * t * f)
    return numpy.sin(x * 8 * (strength + 0.125))



def main():
    #plotSignals(lowPassSignal(saw(1,10000,10),0.5))
    #return
    foldA = numpy.linspace(0.0, 1.0, 2)
    folds = [waveFoldSignal(a, SR=1024) for a in foldA]
    # wavetables = [saw, square, tri, comb, saw_stack]
    wavetables = [OscFromPartials() for w in folds]
    for w,fold in zip(wavetables, folds):
        w.ReconstructFromSignal(fold)
    for (w, a) in zip(wavetables, foldA):
        w.setName("fold_" + str(a))

    freqHarmonics(wavetables)
    # freqSweep(wavetables)
    # plotWav(r'input_fin.wav')
    # plotSignals(waveFoldSignal(1), waveFoldSignal(0.5), waveFoldSignal(0))
    # plotSignals(waveFoldSignal(1))

    # test = WavFile(r'input_fin.wav')
    # findPartialsFromPeaks(test.left, test.sampleRate)


main()
