import numpy as np
import math
import scipy
import scipy.constants
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

states = np.genfromtxt(
    './9Be-3H__Darby-Lewis.states',
    dtype = None,
    names = ('i', 'E', 'gi', 'J', '+/-', 'e/f', 'sigma', 'v', 'b', 'c', 'd'),
    encoding = 'utf-8'
)

trans = np.genfromtxt(
    './9Be-3H__Darby-Lewis.trans',
    dtype = None,
    names = ('i', 'f', 'Afi', 'Vfi'),
    encoding = 'utf-8'
)

in_y = np.genfromtxt(
    './BeT-3.dat',
    dtype = None,
    names = ('lmbd', 'I'),
    encoding = 'utf-8',
)

v_min = {}
in_data = []

for t in trans:
    state = states[t['i'] - 1]
    if state['sigma'] != 'A2Pi':
        continue

    v = state['v']
    Eo = v_min.get(v, None)
    if Eo is None:
        v_arr = states[states['v'] == v]
        Eo = np.min(v_arr['E'])
        v_min[v] = Eo
    Er = state['E'] - Eo

    row = (t['i'], t['f'], t['Afi'], t['Vfi'], state['E'], state['gi'], state['J'], state['f0'],
           state['ef'], state['sigma'], state['v'], state['b'], state['c'], state['d'], Eo, Er)
    in_data.append(row)

in_sp = np.array(in_data, dtype=[('i', '<i8'), ('f', '<i8'), ('Afi', '<f8'), ('Vfi', '<f8'), ('E', '<f8'),
                                 ('gi', '<i8'),('J', '<f8'), ('f0', '<U1'), ('ef', '<U1'), ('sigma', '<U8'),
                                 ('v', '<i8'), ('b', '<i8'), ('c', '<f8'), ('d', '<f8'), ('Eo', '<f8'), ('Er', '<f8')])

def Spectrum(A, Ak, Gi, Eo, Er, Tr, To, Vk, x, shift):
    def G(x, Vk):
        def mu(Vk):
            return shift + 1e7 / Vk
        g_nm = 0.014
        g_m = g_nm * 1e-9
        return np.exp(-0.5 * ((x - mu(Vk)) / g_nm) ** 2) / (g_m * np.sqrt(2 * math.pi))

    return A * np.sum(
            Gi * Ak * np.exp((-Er * 1.99 * 1e-23) / (Tr * scipy.constants.Boltzmann))
            *
            G(x, Vk)
            *
            np.exp((-Eo * 1.99 * 1e-23) / (To * scipy.constants.Boltzmann))
            /
            (Tr * To * scipy.constants.Boltzmann ** 2)
            ,  axis=1) / 1e50


def Spectrum_fit(x, A, Tr, To, shift):
    x_column = x.reshape(-1, 1)
    return Spectrum(A, in_sp['Afi'], in_sp['gi'], in_sp['Eo'], in_sp['Er'], Tr, To, in_sp['Vfi'], x_column, shift)

popt, pcov = curve_fit(Spectrum_fit, in_y['lmbd'], in_y['I'], p0=(1, 4000, 4000, -0.2))

print('Parameters A, To, Tr, mu')
print(popt)
print('Covariance Matrix')
print(pcov)

print('Error standart deviation')
err = np.sqrt(np.diag(pcov))
print(err)


x = in_y['lmbd'].reshape(-1, 1)
y_arr = Spectrum(popt[0], in_sp['Afi'], in_sp['gi'], in_sp['Eo'], in_sp['Er'], popt[1], popt[2], in_sp['Vfi'], x, popt[3])

plt.plot(in_y['lmbd'], in_y['I'], label="input")
plt.plot(in_y['lmbd'], y_arr, label=f"A: {popt[0]} Tr: {popt[1]} To: {popt[2]} shift: {popt[3]}", linestyle='--', )
plt.xlabel('Długość fali [nm]')
plt.ylabel('Natężenie promieniowania [W/sr]')
plt.legend()
plt.show()
