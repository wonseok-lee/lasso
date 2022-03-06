import time
import numpy as np
from util import Record

def soft_threshold(X, rho):
    '''
    :param X: Data Matrix
    :param rho: lamda
    :return: soft_threshold
    '''
    zero = np.matrix(np.zeros(X.shape))
    return np.multiply(np.sign(X), np.maximum(np.abs(X)-rho, zero))

def objective_function(X, beta, y, lamda):
    '''
    :param X: Data matrix
    :param beta: beta
    :param y: target value
    :param lamda: penalty value
    :return: 1/2 * ||y-X*beta||_2^2 + lamda * ||beta||_1
    '''
    return 1/2*np.linalg.norm(X.dot(beta)-y)**2+lamda*np.linalg.norm(beta,1)

def coordinate(X, y, lamda, iter_num):
    '''
    coordinate descent for lasso regression (L2 penalized Regression)

    :param X: Data matrix
    :param beta: initial beta
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    X = (X - np.mean(X, 0)) / (np.std(X, 0))
    X = X / (np.linalg.norm(X, axis = 0))
    _, k = X.shape
    beta = np.zeros((k,1))
    record = Record()
    t = time.time()
    for _ in range(iter_num):

        # update beta
        for j in range(k):
            X_j = X[:, j].reshape(-1, 1)
            y_pred = X.dot(beta)
            beta_j = beta[j]
            beta_j = np.matmul(X_j.T, (y - y_pred + beta_j * X_j))
            beta[j] = soft_threshold(beta_j, lamda)

        obj_temp = objective_function(X, beta, y, lamda)
        loss_temp = np.linalg.norm(y - X.dot(beta)) ** 2
        record.add_obj(obj_temp)
        record.add_loss(loss_temp)
        # record.add_beta(beta)

    t = time.time() - t
    record.add_time(t)
    return record

def proximal(X,y,lamda,iter_num):
    '''
    proximal descent for lasso regression (L2 penalized Regression)

    :param X: Data matrix
    :param beta: initial beta
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    X = (X - np.mean(X,0))/ (np.std(X,0))
    X = X / (np.linalg.norm(X, axis=0))
    beta = np.zeros((X.shape[1],1))
    c = np.linalg.norm(X) ** 2
    record = Record()
    t = time.time()
    for j in range(iter_num):

        # update beta
        beta = soft_threshold(beta - np.matmul(X.T, X.dot(beta) - y) / c, lamda / c)

        obj_temp = objective_function(X,beta,y,lamda)
        loss_temp = np.linalg.norm(y-X.dot(beta)) ** 2
        record.add_obj(obj_temp)
        record.add_loss(loss_temp)
        # record.add_beta(beta)
    t = time.time() - t
    record.add_time(t)
    return record

def acc_proximal1(X,y,lamda,iter_num):
    '''
    accelerated proximal descent for lasso regression (L2 penalized Regression)
    momentum = (j - 2) / (j + 1) * (beta_j - beta_{j-1})

    :param X: Data matrix
    :param beta: initial beta
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    X = (X - np.mean(X, 0)) / (np.std(X, 0))
    X = X / (np.linalg.norm(X, axis=0))
    beta = np.zeros((X.shape[1], 1))
    beta_old = beta.copy()
    c = np.linalg.norm(X) ** 2
    record = Record()
    t = time.time()
    for j in range(1,iter_num+1):
        # update momentum
        beta = beta + (j - 2) / (j + 1) * (beta - beta_old)
        beta = beta - lamda * np.matmul(X.T, (X.dot(beta) - y))

        # update beta
        beta_new = soft_threshold(beta / c, lamda / c)

        obj_temp = objective_function(X, beta_new, y, lamda)
        loss_temp = np.linalg.norm(y - X.dot(beta_new)) ** 2
        record.add_obj(obj_temp)
        record.add_loss(loss_temp)
        # record.add_beta(beta)
        beta, beta_old = beta_new, beta
    t = time.time() - t
    record.add_time(t)
    return record

def acc_proximal2(X,y,lamda,iter_num):
    '''
    accelerated proximal descent for lasso regression (L2 penalized Regression)
    momentum = sqrt((1 + 4*(j**2)))/2

    :param X: Data matrix
    :param beta: initial beta
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    X = (X - np.mean(X, 0)) / (np.std(X, 0))
    X = X / (np.linalg.norm(X, axis=0))
    beta = np.zeros((X.shape[1], 1))
    c = np.linalg.norm(X) ** 2
    k = 1
    record = Record()
    z = beta.copy()
    t = time.time()
    for j in range(1,iter_num+1):
        beta_old = beta.copy()

        # update beta
        z = z + np.matmul(X.T, (y - np.matmul(X, z))) / c
        beta = soft_threshold(z, lamda / c)

        # update momentum
        k0 = k
        k = (1 + np.sqrt(1 + 4 * k ** 2)) / 2
        z = beta + ((k0 - 1) / k) * (beta - beta_old)

        obj_temp = objective_function(X, beta, y, lamda)
        loss_temp = np.linalg.norm(y - X.dot(beta)) ** 2
        record.add_obj(obj_temp)
        record.add_loss(loss_temp)
        # record.add_beta(beta)

    t = time.time() - t
    record.add_time(t)
    return record

def admm(X,y,lamda,rho,iter_num):
    '''
    ADMM for lasso regression (L2 penalized Regression)

    :param X: Data matrix
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    X = (X - np.mean(X, 0)) / (np.std(X, 0))
    X = X / (np.linalg.norm(X, axis=0))
    n, k = X.shape
    alpha = np.zeros((k, 1))
    omega = np.zeros((k, 1))
    c = np.linalg.norm(X) ** 2
    t = time.time()
    record = Record()
    for j in range(iter_num):

        # update beta, alpha, omega
        beta_new = (np.linalg.inv(X.T.dot(X) + rho * np.eye(k))).dot(X.T.dot(y) + rho * (alpha-omega))
        alpha_new = soft_threshold(beta_new + omega,lamda/c)
        omega_new = omega + beta_new - alpha_new

        obj_temp = objective_function(X, beta_new, y, lamda)
        loss_temp = np.linalg.norm(y - X.dot(beta_new)) ** 2
        record.add_obj(obj_temp)
        record.add_loss(loss_temp)
        # record.add_beta(beta)

        beta, alpha, omega = beta_new, alpha_new, omega_new
    t = time.time() - t
    record.add_time(t)
    return record
