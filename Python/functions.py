import numpy as np


def griewank_func(x):
	x = np.array(x).ravel()
	t1 = np.sum(x**2) / 4000
	t2 = np.prod([np.cos(x[idx] / np.sqrt(idx+1)) for idx in range(0, len(x))])
	return t1 - t2 + 1


def rosenbrock_func(x):
	x = np.array(x).ravel()
	return np.sum([100*(x[idx]**2 - x[idx+1])**2 + (x[idx] - 1)**2 for idx in range(0, len(x)-1)])


def rastrigin_func(x):
	x = np.array(x).ravel()
	return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)


def ackley_func(x):
	x = np.array(x).ravel()
	ndim = len(x)
	t1 = np.sum(x**2)
	t2 = np.sum(np.cos(2*np.pi*x))
	return -20*np.exp(-0.2 * np.sqrt(t1 / ndim)) - np.exp(t2 / ndim) + 20 + np.e


def expanded_griewank_rosenbrock_func(x):
	x = np.array(x).ravel()
	results = [griewank_func(rosenbrock_func([x[idx], x[idx+1]])) for idx in range(0, len(x) - 1)]
	return np.sum(results) + griewank_func(rosenbrock_func([x[-1], x[0]]))


def elliptic_func(x):
	x = np.array(x).ravel()
	ndim = len(x)
	idx = np.arange(0, ndim)
	return np.sum((10**6)**(idx/(ndim-1)) * x**2)


def bent_cigar_func(x):
	x = np.array(x).ravel()
	return x[0]**2 + 10**6 * np.sum(x[1:]**2)


def lunacek_bi_rastrigin_func(x):
	miu0 = 2.5
	d = 1
	z = x - miu0
	ndim = len(x)
	s = 1 - 1.0 / (2 * np.sqrt(ndim + 20) - 8.2)
	miu1 = -np.sqrt((miu0**2 - d)/s)
	temp1 = np.sum((x - miu0)**2)
	temp2 = d*ndim + s*np.sum((x - miu1)**2)
	result1 = min(temp1, temp2)
	return result1 + 10*(ndim - np.sum(np.cos(2*np.pi*z)))


def modified_schwefel_func(x):
	x = np.array(x).ravel()
	ndim = len(x)
	z = x + 4.209687462275036e+002
	return 418.9829 * ndim - np.sum(gz_func(z))


def happy_cat_func(x):
	x = np.array(x).ravel()
	ndim = len(x)
	t1 = np.sum(x)
	t2 = np.sum(x**2)
	return np.abs(t2 - ndim)**0.25 + (0.5*t2 + t1) / ndim + 0.5


def hgbat_func(x):
	x = np.array(x).ravel()
	ndim = len(x)
	t1 = np.sum(x)
	t2 = np.sum(x**2)
	return np.abs(t2**2 - t1**2)**0.5 + (0.5*t2 + t1) / ndim + 0.5


def gz_func(x):
	x = np.array(x).ravel()
	ndim = len(x)
	t1 = (500 - np.mod(x, 500)) * np.sin(np.sqrt(np.abs(500 - np.mod(x, 500)))) - (x - 500)**2/(10000*ndim)
	t2 = (np.mod(np.abs(x), 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(x), 500) - 500))) - (x+500)**2/(10000*ndim)
	t3 = x*np.sin(np.abs(x)**0.5)
	conditions = [x < -500, (-500 <= x) & (x <= 500), x > 500]
	choices = [t2, t3, t1]
	y = np.select(conditions, choices, default=np.nan)
	return y


def rotated_expanded_scaffer_func(x):
	x = np.array(x).ravel()
	results = [scaffer_func([x[idx], x[idx+1]]) for idx in range(0, len(x)-1)]
	return np.sum(results) + scaffer_func([x[-1], x[0]])


def scaffer_func(x):
	x = np.array(x).ravel()
	return 0.5 + (np.sin(np.sqrt(np.sum(x**2)))**2 - 0.5) / (1 + 0.001 * np.sum(x**2))**2


def calculate_weight(x, xichma=1.):
	ndim = len(x)
	weight = 1
	temp = np.sum(x ** 2)
	if temp != 0:
		weight = (1.0 / np.sqrt(temp)) * np.exp(-temp / (2 * ndim * xichma ** 2))
	return weight


def discus_func(x):
    x = np.array(x).ravel()
    return 10**6 * x[0]**2 + np.sum(x[1:]**2)
