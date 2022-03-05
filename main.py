from fit import *
from sklearn import datasets
# import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target.reshape(-1,1)

n,k = X.shape
beta = np.ones((k,1))


if __name__=='__main__':
    proximal_descent = proximal(X, beta, y, 0.5, 100)
    acc_proximal_descent1 = acc_proximal1(X, beta, y, 0.5, 100)
    acc_proximal_descent2 = acc_proximal2(X, beta, y, 0.5, 100)
    admm_descent = admm(X, y, 0.5, 100, 100)
    coordinate_descent = coordinate(X, beta, y, 0.5, 100)

    print(f'the final loss of proximal_descent is {round(proximal_descent.loss[-1],4)}')
    print(f'the final loss of acc_proximal_descent1 is {round(acc_proximal_descent1.loss[-1],4)}')
    print(f'the final loss of acc_proximal_descent2 is {round(acc_proximal_descent2.loss[-1],4)}')
    print(f'the final loss of admm_descent is {round(admm_descent.loss[-1],4)}')
    print(f'the final loss of coordinate_descent is {round(coordinate_descent.loss[-1],4)}')
    print('\n')
    print(f'proximal descent took {round(proximal_descent.get_time(),4)}')
    print(f'accelerated proximal descent1 took {round(acc_proximal_descent1.get_time(),4)}')
    print(f'accelerated proximal descent2 took {round(acc_proximal_descent2.get_time(),4)}')
    print(f'admm descent took {round(admm_descent.get_time(),4)}')
    print(f'coordinate_descent took {round(coordinate_descent.get_time(),4)}')
