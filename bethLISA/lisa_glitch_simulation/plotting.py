import numpy as np
import matplotlib.pyplot as plt


def plot_all_four(obs, glitch_times_inject, glitch_times_trigger=None, xrng=1000, yrng=None):

    obsX, obsY, obsZ, obsT = obs['X'], obs['Y'], obs['Z'], obs['T']

    for i in range(len(glitch_times_inject)):

        g_time = glitch_times_inject[i]

        fig, ((axs0, axs1), (axs2, axs3)) = plt.subplots(2, 2)
        plt.suptitle("glitch time: {:.0f}".format(g_time))

        max_valX = np.max(obsX[(np.asarray(obsX.times) > g_time - xrng) & (np.asarray(obsX.times) < g_time + xrng)])
        min_valX = np.min(obsX[(np.asarray(obsX.times) > g_time - xrng) & (np.asarray(obsX.times) < g_time + xrng)])
        max_valY = np.max(obsY[(np.asarray(obsY.times) > g_time - xrng) & (np.asarray(obsY.times) < g_time + xrng)])
        min_valY = np.min(obsY[(np.asarray(obsY.times) > g_time - xrng) & (np.asarray(obsY.times) < g_time + xrng)])
        max_valZ = np.max(obsZ[(np.asarray(obsZ.times) > g_time - xrng) & (np.asarray(obsZ.times) < g_time + xrng)])
        min_valZ = np.min(obsZ[(np.asarray(obsZ.times) > g_time - xrng) & (np.asarray(obsZ.times) < g_time + xrng)])
        max_valT = np.max(obsT[(np.asarray(obsT.times) > g_time - xrng) & (np.asarray(obsT.times) < g_time + xrng)])
        min_valT = np.min(obsT[(np.asarray(obsT.times) > g_time - xrng) & (np.asarray(obsT.times) < g_time + xrng)])

        xlims = (g_time - xrng, g_time + xrng)

        if glitch_times_trigger is None:

            axs0.plot(obsX.times, obsX)
            axs0.axvline(x=g_time, linestyle="--", color='r', alpha=0.5)
            axs0.set_title("TDI-X")
            axs0.set_xlabel("time")
            axs0.set_xlim(xlims)
            axs0.set_ylim((min_valX - 10 ** -20, max_valX + 10 ** -20))

            axs1.plot(obsY.times, obsY)
            axs1.axvline(x=g_time, linestyle="--", color='r', alpha=0.5)
            axs1.set_title("TDI-Y")
            axs1.set_xlabel("time")
            axs1.set_xlim(xlims)
            axs1.set_ylim((min_valY - 10 ** -20, max_valY + 10 ** -20))

            axs2.plot(obsZ.times, obsZ)
            axs2.axvline(x=g_time, linestyle="--", color='r', alpha=0.5)
            axs2.set_title("TDI-Z")
            axs2.set_xlabel("time")
            axs2.set_xlim(xlims)
            axs2.set_ylim((min_valZ - 10 ** -20, max_valZ + 10 ** -20))

            axs3.plot(obsT.times, obsT)
            axs3.axvline(x=g_time, linestyle="--", color='r', alpha=0.5)
            axs3.set_title("TDI-T")
            axs3.set_xlabel("time")
            axs3.set_xlim(xlims)
            axs3.set_ylim((min_valT - 10 ** -20, max_valT + 10 ** -20))

        else:

            g_trigg = glitch_times_trigger[i]
            axs0.plot(obsX.times, obsX)
            axs0.axvline(x=g_time, linestyle="-", color='g', alpha=0.5)
            axs0.axvline(x=g_trigg, linestyle="--", color='r', alpha=0.5)
            axs0.set_title("TDI-X")
            axs0.set_xlabel("time")
            axs0.set_xlim(xlims)
            axs0.set_ylim((min_valX - 10 ** -20, max_valX + 10 ** -20))

            axs1.plot(obsY.times, obsY)
            axs1.axvline(x=g_time, linestyle="-", color='g', alpha=0.5)
            axs1.axvline(x=g_trigg, linestyle="--", color='r', alpha=0.5)
            axs1.set_title("TDI-Y")
            axs1.set_xlabel("time")
            axs1.set_xlim(xlims)
            axs1.set_ylim((min_valY - 10 ** -20, max_valY + 10 ** -20))

            axs2.plot(obsZ.times, obsZ)
            axs2.axvline(x=g_time, linestyle="-", color='g', alpha=0.5)
            axs2.axvline(x=g_trigg, linestyle="--", color='r', alpha=0.5)
            axs2.set_title("TDI-Z")
            axs2.set_xlabel("time")
            axs2.set_xlim(xlims)
            axs2.set_ylim((min_valZ - 10 ** -20, max_valZ + 10 ** -20))

            axs3.plot(obsT.times, obsT)
            axs3.axvline(x=g_time, linestyle="-", color='g', alpha=0.5)
            axs3.axvline(x=g_trigg, linestyle="--", color='r', alpha=0.5)
            axs3.set_title("TDI-T")
            axs3.set_xlabel("time")
            axs3.set_xlim(xlims)
            axs3.set_ylim((min_valT - 10 ** -20, max_valT + 10 ** -20))

        plt.tight_layout()
        plt.show()

