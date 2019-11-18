import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.tsa.arima_process import arma_generate_sample

"""
class ChangeFinder:
    def __init__(self, k, d, r, T1, T2, seed=None):
        np.random.seed(seed)
        self.k = k
        self.d = d
        self.r = r
        self.T1 = T1
        self.T2 = T2
        
        self.mu = np.random.random(d)
        #self.sigma = np.random.random((d, d))
        self.sigma = np.eye(d)

        self.C = np.random.random((k+1, d, d))

        self.past_data = []
        self.scores_1st = []
        self.scores_2nd = []
        
    def update(self, X, T1, T2):
        if len(self.past_data) < self.k:
            self.past_data.append(X)
            return

        self.mu += -self.r * self.mu + self.r * X
        self.C[0, :, :] += -self.r * self.C[0, :, :] + self.r * np.outer(X - self.mu, X - self.mu)
        for i in range(1, self.k+1):
            self.C[i, :, :] += -self.r * self.C[i, :, :] + self.r * np.outer(X - self.mu, self.past_data[-i] - self.mu)

        # Yule-Walker 
        A = self._yule_walker()

        # update
        X_hat = self.mu
        for i in range(len(self.past_data)):
            X_hat += np.dot(A[i], self.past_data[-i-1] - self.mu)
        self.sigma += -self.r * self.sigma + self.r * np.outer(X - X_hat, X - X_hat)
        #print(self.mu)
        #print(self.sigma)

        self.past_data.append(X)
        self.past_data.pop(0)

        score = self.d/2 * np.log(2*np.pi) + 0.5 * np.log(np.linalg.det(self.sigma)) + \
                0.5 * ((X - X_hat).reshape(1, -1).dot(self.sigma).dot((X - X_hat).reshape(-1, 1)))
        self.scores_1st.append(score)
"""

class ChangeFinder:
    def __init__(self, k1, k2, d, r, T1, T2, seed=None):
        np.random.seed(seed)
        self.k1 = k1
        self.k2 = k2
        self.d = d
        self.r = r
        self.T1 = T1
        self.T2 = T2

        self.sdar_1st = SDAR(k1, d, r, T1)
        self.sdar_2nd = SDAR(k2, 1, r, T2)

        self.scores_1st = [np.nan] * k1
        self.scores_2nd = [np.nan] * k2

        self.ma_scores_1st = [np.nan] * (k1+T1)
        self.ma_scores_2nd = [np.nan] * (k2+T1+T2)

        self.t = 0

    def update(self, X):
        self.t += 1
        #if self.t < self.k:
        #    self.past_data.append(X)
        #    return

        score1 = self.sdar_1st.update(X)
        if self.t < self.k1 + self.T1:
            return np.nan
        else:
            self.scores_1st.append(score1[0])
            #print(self.scores_1st[-self.T1:])
            #print(score1)
            mean_score1 = np.mean(np.hstack((self.scores_1st[(-self.T1+1):], score1.tolist())))
            self.ma_scores_1st.append(mean_score1)
            score2 = self.sdar_2nd.update(score1[0])
            #print(score2)

            if self.t >= self.k1 + self.T1 + self.T2:
                self.scores_2nd.append(score2[0])
                mean_score2 = np.mean(np.hstack((self.scores_2nd[(-self.T2+1):], score2.tolist())))
                self.ma_scores_2nd.append(mean_score2)
                return mean_score2
            else:
                return np.nan

    def scores(self):
        return self.ma_scores_2nd


class SDAR:
    def __init__(self, k, d, r, T, seed=None):
        self.k = k
        self.d = d
        self.r = r
        self.T = T

        self.mu = np.random.random(d)
        #self.sigma = np.random.random((d, d))
        self.sigma = np.eye(d)

        self.C = np.random.random((k+1, d, d))
        self.past_data = []

    def update(self, X):
        if len(self.past_data) < self.k:
            self.past_data.append(X)
            return

        self.mu += -self.r * self.mu + self.r * X
        self.C[0, :, :] += -self.r * self.C[0, :, :] + self.r * np.outer(X - self.mu, X - self.mu)
        for i in range(1, self.k+1):
            self.C[i, :, :] += -self.r * self.C[i, :, :] + self.r * np.outer(X - self.mu, self.past_data[-i] - self.mu)

        # Yule-Walker
        A = self._yule_walker()

        # update
        X_hat = self.mu
        for i in range(len(self.past_data)):
            X_hat += np.dot(A[i], self.past_data[-i-1] - self.mu)
        self.sigma += -self.r * self.sigma + self.r * np.outer(X - X_hat, X - X_hat)
        #print(self.mu)
        #print(self.sigma)

        self.past_data.append(X)
        self.past_data.pop(0)

        score = self.d/2 * np.log(2*np.pi) + 0.5 * np.log(np.linalg.slogdet(self.sigma)[1]) + \
                0.5 * ((X - X_hat).reshape(1, -1).dot(np.linalg.inv(self.sigma)).dot((X - X_hat).reshape(-1, 1)))

        return score.ravel()


    def _yule_walker(self):
        """
        solve Yule-Walker equation for VAR model
        :return:
        """
        #V = np.eye(self.d)
        #U = np.eye(self.d)
        #C0 = np.eye(self.d)
        V = self.C[0, :, :].copy()
        U = self.C[0, :, :].copy()
        C0 = self.C[0, :, :].copy()

        A = [np.eye(self.d)]
        B = [np.eye(self.d)]

        for i in range(1, self.k+1):
            #print('i =', i)
            A_i = []
            B_i = []

            # W_{m} = C_{m} - \sum_{i=1}^{m-1} A_{i}^{m-1} C_{m-i}
            W_i = self.C[i, :, :]
            for j in range(i):
                #W_i -= np.dot(A[-j], self.C[i-j, :, :])
                W_i -= np.dot(A[j], self.C[i-j, :, :])

            # A_{m}^{m} = W_{m} U_{m-1}^{-1}
            # B_{m}^{m} = W_{m}^{T} V_{m-1}^{-1}
            A_i_i = np.dot(W_i, np.linalg.inv(U))
            #A_i.append(A_i_i)
            B_i_i = np.dot(W_i.T, np.linalg.inv(V))
            #B_i.append(B_i_i)

            # A_{i}^{m} = A_{i}^{m-1} - A_{m}^{m} B_{m-i}^{m-1}
            # B_{i}^{m} = B_{i}^{m-1} - B_{m}^{m} A_{m-i}^{m-1}
            for j in range(len(A)):
                #A_i_j = A[j] - np.dot(A_i_i, B[i-j])
                #B_i_j = B[j] - np.dot(B_i_i, A[i-j])
                A_i_j = A[j]
                B_i_j = B[j]
                if i >= 1:
                    A_i_j -= np.dot(A_i_i, B[i-1-j])
                    B_i_j -= np.dot(B_i_i, A[i-1-j])
                #else:
                #    A_i_j -= A_i_i
                #    B_i_j -= B_i_i
                A_i.append(A_i_j)
                B_i.append(B_i_j)

            A_i.append(A_i_i)
            B_i.append(B_i_i)

            V = C0.copy()
            U = C0.copy()
            for j in range(i):
                V -= np.dot(A_i[j], self.C[j+1, :, :].T)
                U -= np.dot(B_i[j], self.C[j+1, :, :])

            A = A_i.copy()
            B = B_i.copy()

        return A


if __name__ == '__main__':
    k, d, r, seed = 2, 1, 0.05, 123
    T1, T2 = 5, 7
    
    cf = ChangeFinder(k, d, r, T1, T2, seed)
    
    #X1 = multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).rvs(10000)
    #X2 = multivariate_normal([1.0, 1.0], [[1.0, 0.8], [0.8, 1.0]]).rvs(1000)
    
    #X = np.vstack((X1, X2))
    #X = np.vstack((X1))

    X = arma_generate_sample([1.0, -0.6, 0.5], [1.0, 0.0, 0.0], 100000)
    X = X.reshape(-1, 1)

    for i in range(X.shape[0]):
        cf.update(X[i, :])