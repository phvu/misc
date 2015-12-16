import numpy as np
import matplotlib.pyplot as plt
from gaussian_process import GaussianProcess


def func(x):
    return np.sin(0.9 * x)


g = GaussianProcess(noise_variance=1E-5)


def sample_and_plot():
    xtest = np.linspace(-5, 5, 100).reshape(-1, 1)
    ytest = func(xtest)
    mu, cov_posterior, s = g.predict(xtest)

    plt.figure()
    plt.clf()
    plt.plot(g.X, g.Y, 'r+', ms=20)
    plt.plot(xtest, ytest, 'b-')
    plt.gca().fill_between(xtest.flat, mu[:, 0] - 3*s, mu[:, 0] + 3*s, color="#dddddd")
    plt.plot(xtest, mu, 'r--', lw=2)
    # plt.savefig('predictive.png', bbox_inches='tight')
    plt.title('Mean predictions plus 3 st.deviations')
    plt.axis([-5, 5, -3, 3])

    # draw samples from the posterior at our test points.
    L = np.linalg.cholesky(cov_posterior)
    f_post = mu + np.dot(L, np.random.normal(size=(xtest.shape[0], 10)))
    plt.figure()
    plt.clf()
    plt.plot(xtest, f_post)
    plt.title('Ten samples from the GP posterior')
    plt.axis([-5, 5, -3, 3])
    # plt.savefig('post.png', bbox_inches='tight')

sample_and_plot()

x = np.random.uniform(-5, 5, 10).reshape(-1, 1)
g.predict(x, func(x))
sample_and_plot()

x = np.random.uniform(-5, 5, 10).reshape(-1, 1)
g.predict(x, func(x))
sample_and_plot()

plt.show()
