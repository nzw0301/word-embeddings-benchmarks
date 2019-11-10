"""
 Classes and function for answering analogy questions
"""

import logging

import numpy as np
import sklearn
from six.moves import range

logger = logging.getLogger(__name__)


class SimpleAnalogySolver(sklearn.base.BaseEstimator):
    """
    Answer analogy questions

    Parameters
    ----------
    w : Embedding instance

    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings" O. Levy et al. 2014.

    batch_size : int
      Batch size to use while computing accuracy. This is because of extensive memory usage.

    k: int
      If not None will select k top most frequent words from embedding before doing analogy prediction
      (this can offer significant speedups)

    Note
    ----
    It is suggested to normalize and standardize embedding before passing it to SimpleAnalogySolver.
    To speed up code consider installing OpenBLAS and setting OMP_NUM_THREADS.
    """

    def __init__(self, w, method="add", batch_size=300, k=None):
        self.w = w
        self.batch_size = batch_size
        self.method = method
        self.k = k

    def score(self, X, y):
        """
        Calculate accuracy on analogy questions dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        y : array-like, shape (n_samples, )
          Analogy answers.

        Returns
        -------
        acc : float
          Accuracy
        """
        return np.mean(y == self.predict(X))

    def predict(self, X):
        """
        Answer analogy questions

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        Returns
        -------
        y_pred : array-like, shape (n_samples, )
          Predicted words.
        """
        w = self.w.most_frequent(self.k) if self.k else self.w
        words = self.w.vocabulary.words
        word_id = self.w.vocabulary.word_id
        mean_vector = np.mean(w.vectors, axis=0)
        output = []

        missing_words = 0
        for query in X:
            for query_word in query:
                if query_word not in word_id:
                    missing_words += 1
        if missing_words > 0:
            logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))

        # Batch due to memory constraints (in dot operation)
        normalized_vectors = w.normalize_words().vectors
        for id_batch, start_batch_index in enumerate(range(0, len(X), self.batch_size)):
            end_batch_index = min(start_batch_index + self.batch_size, len(X))
            ids = np.arange(start_batch_index, end_batch_index)
            X_b = X[ids]
            if id_batch % np.floor(len(X) / (10. * self.batch_size)) == 0:
                logger.info("Processing {}/{} batch".format(int(np.ceil(ids[1] / float(self.batch_size))),
                                                            int(np.ceil(X.shape[0] / float(self.batch_size)))))

            A, B, C = np.vstack(w.get(word, mean_vector) for word in X_b[:, 0]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 1]), \
                      np.vstack(w.get(word, mean_vector) for word in X_b[:, 2])

            if self.method == "add":
                A = (A.T / np.linalg.norm(A, ord=2, axis=1)).T
                B = (B.T / np.linalg.norm(B, ord=2, axis=1)).T
                C = (C.T / np.linalg.norm(C, ord=2, axis=1)).T
                X = (B - A + C)
                X = (X.T / np.linalg.norm(X, ord=2, axis=1))
                D = np.dot(normalized_vectors, X)
            elif self.method == "mul":
                D_A = np.log((1.0 + np.dot(w.vectors, A.T)) / 2.0 + 1e-5)
                D_B = np.log((1.0 + np.dot(w.vectors, B.T)) / 2.0 + 1e-5)
                D_C = np.log((1.0 + np.dot(w.vectors, C.T)) / 2.0 + 1e-5)
                D = D_B - D_A + D_C
            else:
                raise RuntimeError("Unrecognized method parameter")

            # Remove words that were originally in the query
            for id, row in enumerate(X_b):
                D[[w.vocabulary.word_id[r] for r in row if r in
                   w.vocabulary.word_id], id] = np.finfo(np.float32).min

            output.append([words[id] for id in D.argmax(axis=0)])

        return np.array([item for sublist in output for item in sublist])
