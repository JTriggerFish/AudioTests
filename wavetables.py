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

WAVETABLE_SIZE = 2047


def sine(frequency):
    t = numpy.arange(0, WAVETABLE_SIZE) / float(WAVETABLE_SIZE - 1)
    if frequency >= WAVETABLE_SIZE / 2:
        return t * 0
    t[-1] = t[0]
    x = numpy.sin(2 * numpy.pi * t * frequency)
    return x


def comb(n):
    x = 0
    for i in xrange(n):
        x += sine(i + 1)
    return x


def tri(n, f=1):
    x = 0
    for i in xrange(int((n-0.01)/2)+1):
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


def saw(nh, f=1):
    x = 0
    for i in xrange(nh):
        if f*(i+1) > nh:
            break
        x += sine((i + 1) * f) / (i + 1)
    return x


def saw_stack(nh, nstack=7):
    x = 0
    for i in xrange(nstack):
        nsubh = min(1+6*i, nh)
        x += saw(nsubh, i + 1) / ((i + 1) ** 0.5)
    return x


def square(n):
    x = 0
    for i in xrange(int((n-0.01)/2)+1):
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
    power = 2.0 ** (1.2  * (power - 0.5))
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
    return 440.0 * pow(2.0, (midiNote - 69)/12)

def makeMipMap(waveFn, splitsPerOctave = None, targetSampleRate = None):
    if splitsPerOctave is None:
        splitsPerOctave = 3
    if targetSampleRate is None:
        targetSampleRate = 44800
    #In Hz
    lowestFreq  = 20.
    #highestFreq = 7040

    numOctaves       = 10
    freqSplits       = [lowestFreq * 2** (i/float(splitsPerOctave)) for i in xrange(0, splitsPerOctave*numOctaves)]
    numHarmonics     = [int(targetSampleRate / 2. / f) for f in freqSplits]
    tables           = map(waveFn, numHarmonics)

    #renormalise
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
    L  = table.size-1 #The last sample repeats the first one
    t  = phase*L
    inext  = int(t) + 1
    i      = inext -1
    return table[i] + (table[inext]-table[i]) * (t-i)

def waveTableOscillator(samples, waveTable, freqFn, sampleRate = None):
    if sampleRate is None:
        sampleRate = 44800

    freqSplits, waves = waveTable

    phase = 0.
    freq = freqFn(samples)
    output = []
    for i in samples:
        f = freq[i]

        tableIdx  = numpy.searchsorted(freqSplits, f) - 1
        tableIdx  = min(tableIdx,len(freqSplits)-1)
        tableNext = min(tableIdx+1,len(freqSplits)-1)

        freq1  = freqSplits[tableIdx]
        freq2  = freqSplits[tableNext]

        phase += f/sampleRate
        phase %= 1.0

        wave1 = interpWave(waves[tableIdx], phase)
        wave2 = interpWave(waves[tableNext], phase)
        v     = wave1 + (f - freq1)/(freq2-freq1) * (wave2-wave1)

        output.append(v)

    return numpy.array(output)


def freqRamp(samples, freqStart, freqStop):
    return numpy.exp(numpy.log(freqStart) + samples * numpy.log(freqStop/freqStart) / (samples.size-1))

def outputFiles(names, soundStreams, sampleRate = None):
    if sampleRate is None:
        sampleRate = 48000
    assert(len(names)==len(soundStreams))
    volume = 1
    numChan = 1 # of channels (1: mono, 2: stereo)
    dataSize = 2 # 2 bytes because of using signed short integers => bit depth = 16

    for name, sound in zip(names, soundStreams):
        data = array.array('h') # signed short integer (-32768 to 32767) data
        N    = sound.size

        for i in range(N):
            sample = 32767 * float(volume)
            sample *= sound[i]
            data.append(int(sample))

        f = wave.open(name+'.wav', 'w')
        f.setparams((numChan, dataSize, sampleRate, N, "NONE", "Uncompressed"))
        f.writeframes(data.tostring())
        f.close()



def freqSweep(wavetables):
    duration = 10 # seconds
    freqStart = 20
    freqStop  = 12000
    sampleRate = 48000
    numSamples = duration * sampleRate
    samples = numpy.arange(numSamples)

    tables = [makeMipMap(w) for w in wavetables]

    names        = ["freqSweep_" + w.__name__ for w in wavetables]
    soundStreams = []

    for wt in tables:
        soundStreams.append(waveTableOscillator(samples, wt, lambda s : freqRamp(s, freqStart, freqStop), sampleRate = sampleRate ))

    outputFiles(names, soundStreams, sampleRate)

def freqHarmonics(wavetables):
    durationPerNote = 2. # seconds
    freqStart = 50.
    sampleRate = 48000
    harmonics = [1,2,3,4,5,6,7,8,9,10,15,20,30,50,100]
    #harmonics = [50]
    numSamples = int(durationPerNote*len(harmonics)*sampleRate)
    samples = numpy.arange(numSamples)

    tables = [makeMipMap(w) for w in wavetables]

    names        = ["freqHarmonics" + w.__name__ for w in wavetables]
    soundStreams = []

    def freqFn():
        freqFn.durationPerNote = durationPerNote
        freqFn.sampleRate = sampleRate
        freqFn.harmonics  = harmonics
        freqFn.freqStart  = freqStart
        def f(samples):
            i = 0
            idx = 0
            freq = []
            for s in samples:
                if i > freqFn.sampleRate * freqFn.durationPerNote:
                    idx += 1
                    i = 0
                idx = min(idx, len(freqFn.harmonics)-1)
                i += 1
                freq.append(freqFn.freqStart*freqFn.harmonics[idx])
            return freq

        def f2(samples):
            return numpy.array([100.0]*samples.size)

        return f

    for wt in tables:
        soundStreams.append(waveTableOscillator(samples, wt, freqFn(), sampleRate = sampleRate ))

    outputFiles(names, soundStreams, sampleRate)

#TEST
#sqTables, sqFreqs = makeMipMap(square)
#exit(0)

#wavetables = [saw, square, tri, comb, saw_stack]
wavetables = [square_formant]

#freqHarmonics(wavetables)
freqSweep(wavetables)



