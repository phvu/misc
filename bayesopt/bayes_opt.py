
def optimize(func, params, max_iters):
    for t in range(0, max_iters):
        # maximizing the acquisition function
        y = func(x)
        # update GP with x, y
