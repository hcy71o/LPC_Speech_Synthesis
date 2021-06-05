from scipy.linalg import solve_toeplitz, toeplitz
import numpy as np


def Autocor(signal, k):

    if k == 0:
        return np.sum(signal**2)
    else:
        return np.sum(signal[k:]*signal[:-k])


def LPF(signal, target, sr):

    # * target: cutoff frequency, sr: sampling_rate
    fft_signal = np.fft.rfft(signal)
    cutoff = int(np.ceil(target/sr*len(fft_signal)))
    fft_signal[cutoff:] = 0
    new_signal = np.fft.irfft(fft_signal)
    return new_signal


def Center_Clip(signal, c_):

    CL = max(abs(signal))*c_  # * 기준

    for i in range(len(signal)):  # * clipping
        if abs(signal[i]) <= CL:
            signal[i] = 0
        elif signal[i] > CL:
            signal[i] -= CL
        elif signal[i] < -CL:
            signal[i] += CL

    return signal


def Pitch_detect(signal, window, sr, overlap=0.5, center_clip=0.68, th_=0.55):

    win_len = len(window)
    overlap_length = int(len(window) * overlap)

    # * zero-padding
    if len(signal) % overlap_length != 0:
        pad = np.zeros(overlap_length - (len(signal) %
                                         overlap_length))  # * 부족한 만큼 0으로 이루어진 배열을 생성
        new_signal = np.append(signal, pad)
    else:
        new_signal = signal

    # * signal이 overlap_length로 몇 번 나눠지는지 구해서, 이후 for문에 사용된다.
    index = (len(new_signal) // overlap_length) - 1
    pitch_contour = np.zeros(index)

    for i in np.arange(index):
        # *windowing
        frame = new_signal[(i*overlap_length)
                            :(i*overlap_length)+win_len] * window

        # * Lowpass filtering - desired cutoff freq: 900Hz (sampling rate: 10kHz)
        filtered_frame = LPF(frame, 900, sr)

        # * Center clipping
        filtered_frame = Center_Clip(filtered_frame, center_clip)

        # * Autocorrelation
        Rn = np.zeros(win_len)
        for k in range(win_len):
            Rn[k] = np.sum(filtered_frame[k:] * filtered_frame[:(win_len-k)])

        # * Voiced/Unvoiced detection
        # * pitch가 나올 구간 (50Hz ~500 Hz) 에서 pitch 위치 찾기
        # * 50Hz: 0.02s, 500 Hz: 0.002s period(s)
        start = int(np.ceil(0.002*sr))
        end = int(np.ceil(0.02*sr))
        max_idx = Rn[start:end].argmax()
        threshold = Rn[0] * th_  # * 임계값 설정
        if Rn[max_idx + start] >= threshold:  # * voiced
            pitch = np.ceil(sr/(max_idx+start))
            pitch_contour[i] = pitch
        else:  # * unvoiced
            pitch_contour[i] = 0

    return pitch_contour


def medianfilter(signal):  # * 5-point median filter
    med_sa = np.zeros((len(signal)+4))
    med_sa[2:-2] = signal  # * 양옆 두 칸 zero-padding
    med_values = []
    for i in range(len(signal)):
        value = sorted(med_sa[i:i+5])[2]
        med_values.append(value)
    # if plot == True:
    #     plt.figure(figsize=(12, 4), dpi=200)
    #     plt.plot(axis, med_values)
    #     plt.xlabel('Time (sec)')
    #     plt.xlim(0, 0.5)
    #     plt.ylabel('Pitch (Hz)')
    #     plt.title('Pitch contour')
    #     plt.grid(True)
    return med_values


def Levinson(w_sig, p):
    r_list = [Autocor(w_sig, i) for i in range(p)]
    b_list = [Autocor(w_sig, i) for i in range(1, p+1)]
    LPC = solve_toeplitz((r_list, r_list), b_list)
    return LPC
