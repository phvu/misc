import matplotlib.pyplot as plt
import numpy as np

plt.xkcd()


def d2():
    plt.figure()

    x = np.arange(0.1, 1, 0.15)
    y = np.arange(0.1, 1, 0.15)
    xx, yy = np.meshgrid(x, y, sparse=False)
    plt.scatter(xx, yy, c='b')
    plt.xlabel('First parameter')
    plt.ylabel('Second parameter')
    plt.title('Grid search')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.figure()
    N = 36
    x = 0.96 * np.random.rand(N) + 0.02
    y = 0.96 * np.random.rand(N) + 0.02

    plt.scatter(x, y, c='b')
    plt.xlabel('First parameter')
    plt.ylabel('Second parameter')
    plt.title('Random search')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()


def d1():
    plt.figure()



if __name__ == '__main__':
    d1()
