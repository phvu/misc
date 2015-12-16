import numpy as np
import matplotlib.pyplot as plt
from gaussian_process import GaussianProcess


def func(v):
    return np.sin(v)


gp = GaussianProcess(noise_variance=1E-5)

fig = plt.figure()
ax_data = fig.add_subplot(211)
ax_func = fig.add_subplot(212)

l_mu = None
l_data = None
l_stddev = None
l_func = None
xx = np.linspace(-5, 5, 100).reshape(-1, 1)


def plot_func(data, mu, cov):
    global l_func

    l = np.linalg.cholesky(cov)
    f_post = mu + np.dot(l, np.random.normal(size=(mu.shape[0], 10)))
    if l_func is not None:
        [a.remove() for a in l_func]
    l_func = ax_func.plot(data, f_post)
    ax_func.axis([-5, 5, -3.5, 3.5])


def onclick(event):
    global l_mu
    global l_data
    global l_stddev

    if event.inaxes != ax_data:
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

    ax_data.axis([-5, 5, -3.5, 3.5])

    # draw the functions
    plot_func(xx, mu, cov)

    fig.canvas.draw()

x = np.linspace(-5, 5, 100).reshape(-1, 1)
y = func(x)

ax_data.plot(x, y, 'b-')
l_mu = ax_data.plot(xx, np.zeros((xx.shape[0], 1)), 'r--', lw=2)[0]
l_stddev = ax_data.fill_between(xx.flat, -3, 3, color='#dddddd')
ax_data.axis([-5, 5, -3.5, 3.5])

plot_func(xx, np.zeros_like(xx), gp.kernel.compute(xx, xx) + gp.noise_variance * np.eye(xx.shape[0]))

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
raw_input()
