"""
https://www.sfu.ca/~ssurjano/optimization.html
"""
import numpy as np


def levy(x: np.ndarray):
    """Levy function

    Description:
    - Dimensions: 2
    - The Levy function is a multimodal function with a large number of local minima.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [1, 1]
    """
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[0]) ** 2 + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)) + (
                w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)


def sphere(x: np.ndarray):
    """Sphere function : https://www.sfu.ca/~ssurjano/spheref.html

    Description:
    - Dimensions: 2
    - The Sphere function is a simple function to optimize, because of its convex shape.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-5.12, 5.12], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [0, 0]
    """
    return np.sum(x ** 2)




def griewank(x: np.ndarray):
    """Griewank function : https://www.sfu.ca/~ssurjano/griewank.html

    Description:
    - Dimensions: 2

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-600, 600], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [0, 0]
    """
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def dixon_price(x: np.ndarray):
    """Dixon-Price function : https://www.sfu.ca/~ssurjano/dixonpr.html

    Description:
    - Dimensions: 2

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [2^(-1/2), 2^(-1/2), ..., 2^(-1/2)]
    """
    return (x[0] - 1) ** 2 + np.sum((2 * np.arange(2, len(x) + 1) - 2) * x[1:] ** 2 + (x[:-1] - 1) ** 2)

def rosenbrock(x: np.ndarray):
    """Rosenbrock function : https://www.sfu.ca/~ssurjano/rosen.html

    Description:
    - Dimensions: d

    Input Domain:
    - It may be evaluated on other domains, but it is most commonly evaluated on the square xi ∈ [-5, 10], for all i = 1, ..., d.

    Global Minimum:
    - f(x*) = 0, at x* = [1, 1, ..., 1]
    """
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin(x: np.ndarray):
    """Rastrigin function : https://www.sfu.ca/~ssurjano/rastr.html

    Description:
    - Dimensions: 2

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-5.12, 5.12], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [0, 0]

    """
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def schwefel(x: np.ndarray):
    """Schwefel function : https://www.sfu.ca/~ssurjano/schwef.html

    Description:
    - Dimensions: 2
    - The Schwefel function is a difficult function to optimize, because of its many local minima.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-500, 500], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [420.9687, 420.9687]
    """
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def zakharov(x: np.ndarray):
    """Zakharov function : https://www.sfu.ca/~ssurjano/zakharov.html

    Description:
    - Dimensions: d

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-5, 10], for all i = 1, ..., d.

    Global Minimum:
    - f(x*) = 0, at x* = [0, 0]
    """
    return np.sum(x ** 2) + np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2 + np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4

def booth(x: np.ndarray):
    """Booth function : https://www.sfu.ca/~ssurjano/booth.html

    Description:
    - Dimensions:

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [1, 3]
    """
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def three_hump_camel(x: np.ndarray):
    """Three-hump camel function : https://www.sfu.ca/~ssurjano/camel3.html

    Description:
    - Dimensions: 2

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-5, 5], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [0, 0]
    """
    return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2


def beale(x: np.ndarray):
    """Beale function : https://www.sfu.ca/~ssurjano/beale.html

    Description:
    - Dimensions: 2
    - The Beale function is a simple function to optimize, because of its convex shape.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-4.5, 4.5], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [3, 0.5]
    """
    return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
                2.625 - x[0] + x[0] * x[1] ** 3) ** 2


def matyas(x: np.ndarray):
    """Matyas function : https://www.sfu.ca/~ssurjano/matyas.html

    Description:
    - Dimensions: 2
    - The Matyas function is a simple function to optimize, because of its convex shape.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 0, at x* = [0, 0]
    """
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


def mccormick(x: np.ndarray):
    """McCormick function : https://www.sfu.ca/~ssurjano/mccorm.html

    Description:
    - Dimensions: 2
    - The McCormick function is a simple function to optimize, because of its convex shape.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-1.5, 4], for all i = 1, 2.

    Global Minimum:
    - f(x*) = -1.9133, at x* = [-0.54719, -1.54719]
    """
    return np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1


def goldstein_price(x: np.ndarray):
    """Goldstein-Price function : https://www.sfu.ca/~ssurjano/goldpr.html

    Description:
    - Dimensions: 2
    - The Goldstein-Price function is a difficult function to optimize, because of its many local minima.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-2, 2], for all i = 1, 2.

    Global Minimum:
    - f(x*) = 3, at x* = [0, -1]
    """
    return (1 + (x[0] + x[1] + 1) ** 2 * (
                19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (
                30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                    18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))


def eggholder(x: np.ndarray):
    """Eggholder function : https://www.sfu.ca/~ssurjano/egg.html

    Description:
    - Dimensions: 2
    - The Eggholder function is a difficult function to optimize, because of its many local minima.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-512, 512], for all i = 1, 2.

    Global Minimum:
    - f(x*) = -959.6407, at x* = [512, 404.2319]
    """
    return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0] / 2 + 47))) - x[0] * np.sin(
        np.sqrt(np.abs(x[0] - (x[1] + 47))))


def cross_in_tray(x: np.ndarray):
    """Cross-in-tray function : https://www.sfu.ca/~ssurjano/crossit.html

    Description:
    - Dimensions: 2
    - The Cross-in-Tray function has multiple global minima. It is shown here with a smaller domain in the second plot, so that its characteristic "cross" will be visible.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.

    Global Minimum:
    - f(x*) = -2.06261, at x* = [1.34941, 1.34941]
    """
    return -0.0001 * (np.abs(
        np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))) + 1) ** 0.1


def drop_wave(x: np.ndarray):
    """Drop-wave function : https://www.sfu.ca/~ssurjano/drop.html

    Description:
    - Dimensions: 2
    - The Drop-wave function is a difficult function to optimize, because of its many local minima.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-5.12, 5.12], for all i = 1, 2.

    Global Minimum:
    - f(x*) = -1, at x* = [0, 0]
    """
    return -(1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))) / (0.5 * (x[0] ** 2 + x[1] ** 2) + 2)

def shubert(x: np.ndarray):
    """Shubert function : https://www.sfu.ca/~ssurjano/shubert.html

    Description:
    - Dimensions: 2
    - The Shubert function is a difficult function to optimize, because of its many local minima.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.

    Global Minimum:
    - f(x*) = -186.7309, at x* = [7.0835, 7.0835]
    """
    return np.sum([i * np.cos((i + 1) * x[0] + i) for i in range(1, 6)]) * np.sum(
        [i * np.cos((i + 1) * x[1] + i) for i in range(1, 6)])

def six_hump_camel_back(x: np.ndarray):
    """Six-hump camel back function : https://www.sfu.ca/~ssurjano/camel6.html

    Description:
    - Dimensions: 2
    - The Six-hump camel back function is a difficult function to optimize, because of its many local minima.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-3, 3], for all i = 1, 2.

    Global Minimum:
    - f(x*) = -1.0316, at x* = [0.0898, -0.7126] or [0.0898, 0.7126]
    """
    return (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 + x[0] * x[1] + (-4 + 4 * x[1] ** 2) * x[1] ** 2

def branin(x : np.ndarray):
    """Branin function : https://www.sfu.ca/~ssurjano/branin.html

    Description:
    - Dimensions: 2
    - The Branin function has 3 global minima.

    Input Domain:
    - The function is usually evaluated on the square x1 ∈ [-5, 10] and x2 ∈ [0, 15].

    Global Minimum:
    - f(x*) = 0.397887, at x* = [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
    """

    # recommanded values
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s



def get_domain_by_function_name(func_name : str, dim : int):
    """"Return the domain of the function (infs : list,  sups : list)"""
    func_name = func_name.lower()
    match func_name:
        case "levy":
            return [-10] * dim, [10] * dim
        case "mccormick":
            return [-1.5] * dim, [4] * dim
        case "eggholder":
            return [-512] * dim, [512] * dim

