import numpy as np
from sklearn.manifold import MDS


def distc(D: int) -> float:
    # correspond to temp[D==1] = 0.5
    if D == 1:
        return 0.5
    gamma = np.arccos(-1/D)
    result = np.sin((np.pi - gamma)/2) / np.sin(gamma)
    return result

def jensen_shannon_divergence(p: np.array, q: np.array):
    m = 0.5 * (p + q)
    # In the R script each sum has na.rm = TRUE as last argument of the function sum
    # Each of the terms is the Kullback–Leibler divergence -> log of the division if the substraction of the logs
    aux = 0.5 * (np.nansum(p * ( np.log2(p) - np.log2(m)))) + 0.5 * (np.nansum(q * ( np.log2(q) - np.log2(m))))
    return aux

def classical_mds(dist_matrix, n_components=2):
    n = len(dist_matrix)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -H.dot(dist_matrix ** 2).dot(H) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs * np.sqrt(eigvals)

def estimateMSVmetrics(probabilities: np.array) -> dict:
    # equivalent to R's ncol()
    n_cols = probabilities.shape[1]
    # creates a matrix full of zeros
    distsM = np.zeros(shape=(n_cols, n_cols))
    # in R indexes are from 1 to N, so 1 to n_cols-1 is 0 to n_cols-2
    # don't do the most-right element
    for i in range(n_cols - 1):
        for j in range(i + 1, n_cols):
            jsdiv = jensen_shannon_divergence(probabilities[:, i], probabilities[:, j])
            distsM[i, j] = np.sqrt(jsdiv)
            # pairwise distance
            distsM[j, i] = distsM[i, j]

    vertices = classical_mds(distsM, n_components=n_cols - 1)
    c = np.sum(vertices, axis=0) / n_cols
    # Repeat the array C to calculate the distance to the center
    cc = np.tile(c, n_cols).reshape((n_cols, -1), order='C')
    cc2 = vertices - cc

    dc = np.linalg.norm(cc2, axis=1)

    gpdmetric = np.mean(dc) / distc(n_cols)
    sposmetrics = dc / (1 - (1 / n_cols))

    return {"GPD": gpdmetric, "SPOs": sposmetrics, "Vertices": vertices}