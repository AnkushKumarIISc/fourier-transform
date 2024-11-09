import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read, write

def dft(x):
    N = len(x)
    X = list()
    for k in range(N):
        X_k = 0
        for n in range(N):
            X_k += x[n] * np.exp(-1j * 2 * np.pi * (k / N) * n)
        X.append(X_k)
    return X

def idft(X):
    N = len(X)
    x = list()
    for n in range(N):
        x_n = 0
        for k in range(N):
            x_n += X[k] * np.exp(1j * 2 * np.pi * (k / N) * n)
        x.append(x_n / N)
    return x

def fft(x):
    return np.fft.fft(x)

def ifft(X):
    return np.fft.ifft(X)

def frequency_separator(X, a, b):
    new_X = X.copy()
    N = len(X)
    for i in range(N):
        if i < int(N / 2):
            if freq_axis[i] < a or freq_axis[i] > b:
                new_X[i] = 0
        else:
            if freq_axis[i] > fs - a or freq_axis[i] < fs - b:
                new_X[i] = 0
    return new_X

"""
x = [1, 2-1j, -1j, -1+2j]
X = dft(x)
y = idft(X)

print(x)
print(X)
print(y)
"""

"""
fs = 2000
T = 0.1
N = int(fs * T)

time_steps = 1 / fs
freq_steps = fs / N

time_axis = np.linspace(0, N * time_steps, N)
freq_axis = np.linspace(0, N * freq_steps, N)

#x = np.sin(2 * np.pi * 100 * time_axis)
#x = np.sin(2 * np.pi * 100 * time_axis) + 2 * np.sin(2 * np.pi * 500 * time_axis) + 3
x = 1 * np.sin(2 * np.pi * 100 * time_axis) + 3 * np.sin(2 * np.pi * 300 * time_axis)

plt.plot(time_axis, x, '.-')
plt.grid()
plt.xlabel('time_axis')
plt.ylabel('x')
plt.title('Time Domain Wave')
plt.show()

X = dft(x)

plot_freq_axis = freq_axis[:int(N / 2)]
plot_X = np.abs(X[:int(N / 2)]) / (N / 2)
plot_X[0] /= 2

plt.plot(plot_freq_axis, plot_X, '.-')
plt.grid()
plt.xlabel('freq_axis')
plt.ylabel('X')
plt.title('Frequency Domain Wave')
plt.show()

for i in range(N):
    if i < int(N / 2):
        if freq_axis[i] > 150:
            X[i] = 0
    else:
        if freq_axis[i] < fs - 150:
            X[i] = 0

y = np.real(idft(X))

plt.plot(time_axis, y, '.-')
plt.grid()
plt.xlabel('time_axis')
plt.ylabel('y')
plt.title('Time Domain Wave')
plt.show()
"""

fs, amplitudes = read('theDOGS.wav')
x0 = amplitudes[:, 0]
x1 = amplitudes[:, 1]
N = len(x0)
print(type(max(x0)))
time_steps = 1 / fs
freq_steps = fs / N

time_axis = np.linspace(0, N * time_steps, N)
freq_axis = np.linspace(0, N * freq_steps, N)

#plt.plot(time_axis, x0, '.-')
#plt.plot(time_axis, x1, '.-')
#plt.grid()
#plt.xlabel('time_axis')
#plt.ylabel('x')
#plt.title('Time Domain Wave')
#plt.show()

X0 = fft(x0)
X1 = fft(x1)

#plot_freq_axis = freq_axis[:int(N / 2)]
#plot_X0 = np.abs(X0[:int(N / 2)]) / (N / 2)
#plot_X1 = np.abs(X1[:int(N / 2)]) / (N / 2)
#plot_X0[0] /= 2
#plot_X1[0] /= 2

#plt.plot(plot_freq_axis, plot_X0, '.-')
#plt.plot(plot_freq_axis, plot_X1, '.-')
#plt.grid()
#plt.xlabel('freq_axis')
#plt.ylabel('X')
#plt.title('Frequency Domain Wave')
#plt.show()

sub_bass_X0 = frequency_separator(X0, 20, 60)
sub_bass_X1 = frequency_separator(X1, 20, 60)
bass_X0 = frequency_separator(X0, 60, 250)
bass_X1 = frequency_separator(X1, 60, 250)
low_midrange_X0 = frequency_separator(X0, 250, 500)
low_midrange_X1 = frequency_separator(X1, 250, 500)
midrange_X0 = frequency_separator(X0, 500, 2000)
midrange_X1 = frequency_separator(X1, 500, 2000)
upper_midrange_X0 = frequency_separator(X0, 2000, 4000)
upper_midrange_X1 = frequency_separator(X1, 2000, 4000)
presence_X0 = frequency_separator(X0, 4000, 6000)
presence_X1 = frequency_separator(X1, 4000, 6000)
brilliance_X0 = frequency_separator(X0, 6000, 20000)
brilliance_X1 = frequency_separator(X1, 6000, 20000)

sub_bass_x0 = ifft(sub_bass_X0)
sub_bass_x1 = ifft(sub_bass_X1)
sub_bass = list()
for i in range(N):
    sub_bass.append([sub_bass_x0[i], sub_bass_x1[i]])
write("0 sub bass.wav", fs, np.array(sub_bass).astype(np.int16))

bass_x0 = ifft(bass_X0)
bass_x1 = ifft(bass_X1)
bass = list()
for i in range(N):
    bass.append([bass_x0[i], bass_x1[i]])
write("1 bass.wav", fs, np.array(bass).astype(np.int16))

low_midrange_x0 = ifft(low_midrange_X0)
low_midrange_x1 = ifft(low_midrange_X1)
low_midrange = list()
for i in range(N):
    low_midrange.append([low_midrange_x0[i], low_midrange_x1[i]])
write("2 low midrange.wav", fs, np.array(low_midrange).astype(np.int16))

midrange_x0 = ifft(midrange_X0)
midrange_x1 = ifft(midrange_X1)
midrange = list()
for i in range(N):
    midrange.append([midrange_x0[i], midrange_x1[i]])
write("3 midrange.wav", fs, np.array(midrange).astype(np.int16))

upper_midrange_x0 = ifft(upper_midrange_X0)
upper_midrange_x1 = ifft(upper_midrange_X1)
upper_midrange = list()
for i in range(N):
    upper_midrange.append([upper_midrange_x0[i], upper_midrange_x1[i]])
write("4 upper midrange.wav", fs, np.array(upper_midrange).astype(np.int16))

presence_x0 = ifft(presence_X0)
presence_x1 = ifft(presence_X1)
presence = list()
for i in range(N):
    presence.append([presence_x0[i], presence_x1[i]])
write("5 presence.wav", fs, np.array(presence).astype(np.int16))

brilliance_x0 = ifft(brilliance_X0)
brilliance_x1 = ifft(brilliance_X1)
brilliance = list()
for i in range(N):
    brilliance.append([brilliance_x0[i], brilliance_x1[i]])
write("6 brilliance.wav", fs, np.array(brilliance).astype(np.int16))
