import numpy as np
from db_model import UserTopic, Topic
import pandas as pd


class Collab_Filtering:

    def __init__(self, session, steps=5000, alpha=0.0002, beta=0.02):
        self.session = session
        self.steps = steps
        self.alpha = alpha
        self.beta = beta

    """
    @INPUT:
        R     : a matrix to be factorized, dimension N x M
        P     : an initial matrix of dimension N x K
        Q     : an initial matrix of dimension M x K
        K     : the number of latent features
        steps : the maximum number of steps to perform the optimisation
        alpha : the learning rate
        beta  : the regularization parameter
    @OUTPUT:
        the final matrices P and Q
    """

    def matrix_factorization(self, R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):

        Q = Q.T
        for step in range(steps):
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                        for k in range(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            eR = np.dot(P, Q)
            e = 0
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                        for k in range(K):
                            e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
            if e < 0.001:
                break
        return P, Q.T

    def prepare_m(self):

        statement = self.session.query(UserTopic.user_id, UserTopic.topic_id, UserTopic.true_score).order_by(
            'user_id, topic_id')
        user_topics = pd.DataFrame(columns=('user_id', 'topic_id', 'true_score'), data=statement.all())

        pivoted = user_topics.pivot(index='user_id', columns='topic_id', values='true_score').reset_index()
        pivoted = pivoted.fillna(0)

        topics = self.session.query(Topic).all()
        total_topics = len(topics)

        z = [[None for j in range(total_topics)] for i in range(pivoted.shape[0])]

        for i in range(len(z)):
            for j in range(len(z[i])):
                z[i][j] = pivoted.iloc[i, j + 1]

        r = np.array(z)

        n = len(r)
        m = len(r[0])
        k = 10

        p = np.random.rand(n, k)
        q = np.random.rand(m, k)

        nP, nQ = self.matrix_factorization(r, p, q, k, self.steps, self.alpha, self.beta)
        nR = np.dot(nP, nQ.T)

        # save the predicted scores to users
        for i in range(len(nR)):
            for j in range(len(nR[i])):
                user_id = int(pivoted.iloc[i, 0])
                topic_id = pivoted.columns[j + 1]
                UserTopic.update_predicted_score(self.session, user_id, topic_id, nR[i][j])

    def run(self):
        self.prepare_m()
