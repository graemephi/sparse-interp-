import re
import struct
import time
import wave
from collections import deque, namedtuple
from pathlib import Path
from random import randint, shuffle

import numpy as np
from scipy import signal as scisig
from sklearn import linear_model as lm

db0 = pow(2, 15)

LearnState = namedtuple('LearnState', ['A', 'B', 't', 'history'])

# src: Directory with training samples. signal length is inferred from the samples.
# words: number of words (atoms) in the dictionary
# reg_scale: tuning parameter. The higher, the more like least-squares the resulting regression is.
# iterations: the more the better
# dest: name for the dictionary and preset files
def create_dictionary(src, words, reg_scale, iterations, dest):
    state = None
    d = sparse.PolarDistribution(src)
    L = d.sample(words).T
    M = None
    for i in range(0, iterations):
        state=sparse.learn(d, L, iters = 4, batch_size = 32, regularisation = reg_scale / (dist.word_len ** .5), max_update_iters=100, state=state)
        if np.isnan(L).any():
            print("Failed")
            break

        replaced = 0
        for i in range(0, L.shape[1]):
            if np.max(np.abs(state.A[i])) < 1e-10:
                L.T[i] = d.sample(1)
                replaced += 1
        if replaced > 0:
            print("replaced %i words" % replaced)
        M = np.copy(L)

    d.dump(M, dest, fitter=lm.LassoLars(alpha = reg_scale / (dist.word_len ** .5)))

# Online Dictionary Learning for Sparse Coding, Julien et al, 2009
def learn(distribution, D, iters = 0, batch_size = 0, regularisation = 1, max_update_iters = 16, state = None):
    start_time = time.perf_counter()

    eps = 3e-5

    history_len = 100

    m = D.shape[0]
    k = D.shape[1]

    if iters == 0:
        iters = 8
    assert(iters > 0)
    if batch_size == 0:
        batch_size = k
    assert(batch_size > 0)

    A = None
    B = None
    t = None
    history = None

    if state is not None:
        A = state.A
        B = state.B
        t = state.t
        history = state.history

    if A is None:
        A = np.zeros((k, k))
    assert(A.shape == (k, k))

    if B is None:
        B = np.zeros_like(D)
    assert(B.shape == D.shape)

    if t is None:
        t = 0
    assert(t >= 0 and t <= history_len)

    if history is None:
        history = deque()
    assert(len(history) <= history_len)

    lars = lm.LassoLars(alpha = regularisation, fit_path=False)

    aa_t = np.zeros_like(A)
    xa_t = np.zeros_like(B)

    failed_convergence_count = 0

    for _ in range(t, t + iters):
        beta = 0.
        theta = 0.

        if t < batch_size:
            theta = t*batch_size
        else:
            theta = batch_size*batch_size + t - batch_size
        beta = (theta + 1 - batch_size) / (batch_size + 1)

        aa_t.fill(0)
        xa_t.fill(0)

        x = distribution.sample(batch_size)

        lars.fit(D, x.T)
        alpha = lars.coef_

        for i in range(0, batch_size):
            aa_t += np.outer(alpha[i], alpha[i])
            xa_t += np.outer(x[i], alpha[i])

        history.append((aa_t, xa_t, beta))

        A = beta*A + aa_t
        B = beta*B + xa_t

        if t == history_len:
            discarded = history.popleft()
            A -= discarded[0]
            B -= discarded[1]
            if discarded[2]:
                A /= discarded[2]
                B /= discarded[2]
        else:
            t += 1
        assert(t <= history_len)

        failed_convergence = False

        # Dictionary update
        for i in range(0, max_update_iters):
            D_prev = np.copy(D)
            prev_test = 10000
            for j in range(0, k):
                if A[j, j] != 0:
                    D.T[j] += ((1 / A[j, j]) * (B.T[j] - np.dot(D, A.T[j])))
                    D.T[j] *= (1 / np.maximum(np.linalg.norm(D.T[j]), 1))

            converge_test = np.max(np.abs(D - D_prev))

            if converge_test < eps:
                break

            if i == (max_update_iters - 1) or prev_test < converge_test:
                failed_convergence = True
                break

            prev_test = converge_test

        if failed_convergence:
            failed_convergence_count += 1

        print(',' if failed_convergence else '.', end = '')

    end_time = time.perf_counter()

    m,s = divmod(end_time - start_time, 60)
    print(" ({:.0f}m {:.3f}s)".format(m,s))

    if failed_convergence_count > 0:
        print("Dictionary update failed to converge {} out of {} times".format(failed_convergence_count, iters))

    return LearnState(A, B, t, history)

def is_power_of_two(n):
    return n != 0 and (n & (n - 1)) == 0

def next_power_of_two(n):
    if is_power_of_two(n):
        return n
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

def normalize(a):
    n = np.linalg.norm(a)
    if n != 0:
        a /= n
    return a

def normalize_rows(A):
    np.apply_along_axis(normalize, 1, A)
    return A

def zeropad(x, new_length):
    n = new_length - x.size
    if n > 0:
        return np.lib.pad(x, (0,n), 'constant')
    return x

def fft_polar(x):
    y = np.fft.rfft(x)
    return polarise(y)

def polarise(x):
    result = np.empty((x.size * 2,))
    result[0::2] = np.absolute(x)
    result[1::2] = np.angle(x)
    return result

def ifft_polar(x):
    y = np.fft.irfft(x[0::2] * np.exp(1j*x[1::2]))
    return y

def random_box_filter(x):
    x = np.copy(x)

    k = np.random.poisson(1)
    if k == 0:
        return x

    if k == 1:
        r = np.random.choice(2)
        m = x.size >> 1
        if r == 0:
            x[:m] = 0
        else :
            x[m:] = 0
    if k >= 2:
        r = np.random.choice(4)
        q = x.size >> 2
        if r == 0:
            x[:q] = 0
        elif r == 1:
            x[q:2*q] = 0
        elif r == 2:
            x[2*q:3*q] = 0
        elif r == 3:
            x[3*q:] = 0

    return x

def random_filter(x, f=np.random.rand(), type=np.random.choice(['lowpass', 'highpass'])):
    if type == 'highpass':
        f = 1 - f
    b, a = scisig.butter(3, f, btype=type)
    y = scisig.filtfilt(b, a, x, method="gust")
    return y


class Urn():
    def __init__(self, size):
        self.size = size
        self.memory = [i for i in range(0, size)]
        shuffle(self.memory)

    def gen_n(self, n):
        return [self.gen() for _ in range(0, n)]

    def gen(self):
        result = self.memory.pop()
        if len(self.memory) == 0:
            self.memory.extend(range(0, self.size))
            shuffle(self.memory)
        return result

# Polar samples are (m, s) for each row and both need to be aligned. so pad, pad, pad
def make_mxe_loadable(X, alignment):
    X = np.require(X, dtype=np.float32, requirements=['C','A'])
    assert(X.flags.c_contiguous)

    mid = X.shape[1] // 2
    off = mid & (alignment - 1)

    new_row_length = mid if off == 0 else (mid & ~(alignment - 1)) + alignment
    assert((new_row_length & (alignment - 1)) == 0)
    n = new_row_length - mid
    if n > 0:
        L = np.lib.pad(X[:, :mid], ((0,0), (0, n)), 'constant', constant_values=np.int32(-1).view(np.float32))
        R = np.lib.pad(X[:, mid:], ((0,0), (0, n)), 'constant', constant_values=np.int32(-1).view(np.float32))
        return np.concatenate((L,R), axis=1)
    return np.copy(X)

def absmax(x):
    return np.max(np.abs(x))

class PoissonDisc1DZ():
    class Draw():
        def __init__(self, value, is_new):
            self.value = value
            self.is_new = is_new

        def __hash__(self):
            return self.value


    def __init__(self, magnitude):
        self.m = magnitude - 1
        self.z = [self.Draw(randint(0, self.m), False)]
        self.z_max = int(magnitude ** .5)
        self.draws = max(int(self.z_max ** .5), 2)
        self.z_idx = 0

    def draw(self):
        x = [self.Draw(randint(0, self.m), True) for _ in range(0, self.draws)]
        zx = sorted(set(self.z).union(set(x)), key=lambda a: a.value)

        furthest = -1
        result = 0
        for i in range(0, len(zx) - 1):
            a = zx[i]
            b = zx[i + 1]
            if a.is_new ^ b.is_new:
                difference = abs(a.value - b.value)
                if difference > furthest:
                    furthest = difference
                    result = a.value if a.is_new else b.value

        assert(furthest != -1)

        if (len(self.z) < self.z_max):
            self.z.append(self.Draw(result, False))
        else:
            self.z[self.z_idx] = self.Draw(result, False)
            self.z_idx += 1
            if self.z_idx == len(self.z):
                self.z_idx = 0

        return result


def tuplise(x, y):
    return (x, y)

class WaveLoader():
    files = {}

    def load_wave(self, file, post_process=tuplise):
        f = str(file)
        cached_wav = self.files.get(f)

        result = None
        if cached_wav != None:
            result = cached_wav
        else:
            with wave.open(f) as wav:
                sample_bytes = wav.readframes(wav.getnframes())
                sample_bytes_size = wav.getnframes() * wav.getnchannels()
                samps = np.array(struct.unpack("<{}h".format(sample_bytes_size), sample_bytes))
                zn = wav.getnframes() * wav.getnchannels()
                assert(wav.getnchannels() == 2)
                if (zn & 1) == 1:
                    zn -= 1
                l = samps[0:zn:2]
                r = samps[1:zn:2]
                mid = (l + r) / db0
                side = (l - r) / db0
                result = post_process(mid, side)
                self.files[f] = result
        assert(result)

        return result

class Distribution():
    pass

class TimeDistribution(Distribution):
    wl = WaveLoader()

    def __init__(self, path, word_len = 0):
        p = Path(path)
        self.files = [str(x) for x in p.iterdir() if x.is_file()]
        self.loaded_files = {}
        self.size = len(self.files)
        self.urn = Urn(self.size)

        with wave.open(str(self.files[0])) as first:
            self.word_len = word_len
            if word_len == 0:
                # All waves must have the same length if its unspecified. Laziness
                self.word_len = first.getnframes() * first.getnchannels()

    def load_sample(self, f, x=None):
        ms = self.wl.load_wave(f)
        s = self.word_len // 2
        if x is None:
            x = np.zeros((self.word_len,))
        f = np.random.rand() * 0.75 + .25
        t = np.random.choice(['lowpass', 'highpass'])
        x[:s] = random_filter(ms[0][:s], f, t)
        x[s:] = random_filter(ms[1][:s], f, t)
        return x

    def sample(self, n):
        assert(n > 0)
        X = np.zeros((n, self.word_len))
        for i in range(0, n):
            self.load_sample(self.files[self.urn.gen()], X[i])
        return X

    def synthesise(self, A, x):
        y = np.dot(A, x)
        n = y.size // 2
        mid = y[:n]
        side = y[n:]
        l = (mid + side) / 2
        r = (mid - side) / 2
        lr = np.stack((l, r))
        return lr

    def dump(self, dictionary, path, fitter=None):
        bases = dictionary.shape[1]
        assert(bases < dictionary.shape[0])
        backup_fitter = lm.OrthogonalMatchingPursuit(4)

        if fitter is None:
            fitter = backup_fitter
        coefs = []

        bad_samples = 0
        for f in self.files:
            x = self.load_sample(f).T
            fitter.fit(dictionary, x)
            b = None
            if absmax(fitter.coef_) != 0:
                b = np.copy(fitter.coef_)
            else:
                backup_fitter.fit(dictionary, x)
                if absmax(backup_fitter.coef_) != 0:
                    b = np.copy(backup_fitter.coef_)

            if b is not None:
                coefs.append(b)
            else:
                bad_samples += 1

        X = np.require(dictionary.T, np.float32, ['A', 'C'])
        Y = np.require(np.array(coefs), np.float32, ['A', 'C'])
        np.save(path, X)
        np.save(path + " coefficients", Y)


# This is criminally stupid but has interesting results
class PolarDistribution(Distribution):
    wl = WaveLoader()

    def __init__(self, path, resample=False, fade=False, word_len_time_mono=None):
        p = Path(path)
        self.files = [str(x) for x in p.iterdir() if x.is_file()]
        self.loaded_files = {}
        self.size = len(self.files)
        self.urn = Urn(self.size)
        self.word_len = 0
        self.resample = resample
        self.fade = fade

        if fade:
            assert(resample == False)

        if word_len_time_mono == None:
            with wave.open(str(self.files[0])) as first:
                assert(first.getnchannels() == 2)
                if fade == True:
                    self.word_len = next_power_of_two(first.getnframes() * 2) // 2 + 4
                elif resample == False:
                    self.word_len = next_power_of_two(first.getnframes() * 2) + 4
                else:
                    # Same length for all wavs assumed!!
                    len_low = next_power_of_two(first.getnframes() * 2) // 2
                    len_high = len_low * 2
                    if abs(first.getnframes()*2 - len_low) < abs(first.getnframes()*2 - len_high):
                        self.word_len = len_low + 4
                    else:
                        self.word_len = len_high + 4

        else:
            self.word_len = word_len_time_mono * 2 + 4
        # mid and side require 2 floats for the complex value at nyquist
        assert(is_power_of_two(self.word_len - 4))

    def load_sample(self, f, x=None):
        def reshape(m, s):
            cols = self.word_len // 2 - 2
            if self.resample:
                m = scisig.resample(m, cols)
                s = scisig.resample(s, cols)
            elif self.fade:
                mm = np.resize(m, (cols,))
                ss = np.resize(s, (cols,))
                # m_tail = m[cols:]
                # s_tail = s[cols:]
                # tail_len = m_tail.shape[0]
                tail_len = 4096
                window = np.hanning(tail_len * 2)
                mm[:tail_len] = mm[:tail_len] * window[:tail_len] # + m_tail * window[tail_len:]
                ss[:tail_len] = ss[:tail_len] * window[:tail_len] # + s_tail * window[tail_len:]
                mm[-tail_len:] = mm[-tail_len:] * window[tail_len:]
                ss[-tail_len:] = ss[-tail_len:] * window[tail_len:]
                m = mm
                s = ss
            else:
                m.resize((cols,), refcheck=False)
                s.resize((cols,), refcheck=False)
            return (fft_polar(m), fft_polar(s))
        ms = self.wl.load_wave(f, reshape)
        s = self.word_len // 2
        if x is None:
            x = np.zeros((self.word_len,))
        x[:s] = ms[0]
        x[s:] = ms[1]
        return x

    def sample(self, n):
        assert(n > 0)
        X = np.zeros((n, self.word_len))
        for i in range(0, n):
            self.load_sample(self.files[self.urn.gen()], X[i])
        return X

    def synthesise(self, A, x):
        y = np.dot(A, x)
        n = y.size // 2
        mid = ifft_polar(y[:n])
        side = ifft_polar(y[n:])
        l = (mid + side) / 2
        r = (mid - side) / 2
        lr = np.stack((l, r))
        return lr

    def dump(self, dictionary, path, fitter=None):
        bases = dictionary.shape[1]
        assert(bases < dictionary.shape[0])
        backup_fitter = lm.OrthogonalMatchingPursuit(4)

        if fitter is None:
            fitter = backup_fitter
        coefs = []

        bad_samples = 0
        for f in self.files:
            x = self.load_sample(f).T
            fitter.fit(dictionary, x)
            b = None
            if absmax(fitter.coef_) != 0:
                b = np.copy(fitter.coef_)
            else:
                backup_fitter.fit(dictionary, x)
                if absmax(backup_fitter.coef_) != 0:
                    b = np.copy(backup_fitter.coef_)

            if b is not None:
                coefs.append(b)
            else:
                bad_samples += 1

        X = make_mxe_loadable(dictionary.T.astype(np.float32), 16)
        Y = np.array(coefs).astype(np.float32)
        np.save(path, X)
        np.save(path + " coefficients", Y)

class SingleDistribution(Distribution):
    def __init__(self, path, word_len):
        self.signal = np.array(WaveLoader().load_wave(path))
        self.signal_len = self.signal.shape[1]
        self.set_word_len(word_len)

    def set_word_len(self, word_len):
        self.word_len = word_len * 2
        self.rng = PoissonDisc1DZ(self.signal_len - word_len)

    def sample(self, n):
        X = np.zeros((n, self.word_len))
        for i in range(0, n):
            s = self.rng.draw()
            x = self.at(s, normalize=True)
            X[i, :] = x
        return X

    def at(self, n, normalize = False):
        mono_len = self.word_len // 2
        x = np.copy(self.signal[:, n : n + mono_len])
        if normalize:
            normalize_rows(x)
        return np.reshape(x, -1)
