import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as stats_norm

from gaussian_process import GaussianProcess, SquaredDistanceKernel, Matern52Kernel


BOUNDS = [0, 10, -10, 10]
PLOT_POINT_COUNT = 1000


def func(v):
    return v * np.sin(v)


# gp = GaussianProcess(kernel=SquaredDistanceKernel(kernel_param=0.01), noise_variance=1E-3)
gp = GaussianProcess(kernel=Matern52Kernel(kernel_param=0.01), noise_variance=1E-3)

fig = plt.figure()
ax_data = fig.add_subplot(311)
ax_acquisition = fig.add_subplot(312)
ax_func = fig.add_subplot(313)


l_mu = None
l_data = None
l_stddev = None
l_acquisition = None
l_acquisition_area = None
l_func = None
xx = np.linspace(BOUNDS[0], BOUNDS[1], PLOT_POINT_COUNT).reshape(-1, 1)


def plot_func(data, mu, cov):
    global l_func

    l = np.linalg.cholesky(cov)
    f_post = mu + np.dot(l, np.random.normal(size=(mu.shape[0], 10)))
    if l_func is not None:
        [a.remove() for a in l_func]
    l_func = ax_func.plot(data, f_post)
    ax_func.axis(BOUNDS)


def onclick(event):
    global l_mu
    global l_data
    global l_stddev
    global l_acquisition
    global l_acquisition_area

    if event.inaxes != ax_data or event.button != 1:
        return

    # update the GP with the new data
    x = np.asarray([[event.xdata]])
    gp.predict(x, func(x))

    # draw the mu and stddev
    mu, cov, s = gp.predict(xx)
    if l_mu is not None:
        l_mu.set_data(xx, mu)
    else:
        l_mu = ax_data.plot(xx, mu, 'r--', lw=2)[0]

    # draw the points
    if l_data is not None:
        l_data.set_data(gp.X, gp.Y)
    else:
        l_data = ax_data.plot(gp.X, gp.Y, 'r+', ms=20)[0]

    # draw the stddev
    if l_stddev is not None:
        l_stddev.remove()
    l_stddev = ax_data.fill_between(xx.flat, mu[:, 0] - 3*s, mu[:, 0] + 3*s, color="#dddddd")

    # draw the functions
    plot_func(xx, mu, cov)

    # probability of improvement (acquisition function)
    eps = 1E-4
    pi = stats_norm.cdf(np.divide(mu - gp.max_observed_value - eps, s[:, None]))

    if l_acquisition is not None:
        l_acquisition.set_data(xx, pi)
    else:
        l_acquisition = ax_acquisition.plot(xx, pi, 'g-', lw=2)[0]

    if l_acquisition_area is not None:
        l_acquisition_area.remove()
    l_acquisition_area = ax_acquisition.fill_between(xx.flat, -1, pi.flat, color="#ccffcc")

    ax_acquisition.axis([BOUNDS[0], BOUNDS[1], -0.5, 1])

    fig.canvas.draw()

x = np.linspace(BOUNDS[0], BOUNDS[1], PLOT_POINT_COUNT).reshape(-1, 1)
y = func(x)

ax_data.plot(x, y, 'b-')
l_mu = ax_data.plot(xx, np.zeros((xx.shape[0], 1)), 'r--', lw=2)[0]
l_stddev = ax_data.fill_between(xx.flat, -3, 3, color='#dddddd')
ax_data.axis(BOUNDS)

ax_acquisition.axis([BOUNDS[0], BOUNDS[1], -0.5, 1])

plot_func(xx, np.zeros_like(xx), gp.kernel.compute(xx, xx) + gp.noise_variance * np.eye(xx.shape[0]))

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
