import os
import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from sklearn import metrics
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class ClusterProcessor:

    def __init__(self, data, sur_data):
        self.data = data
        self.sur_data = sur_data
        self.logrank_cache = {}

    def KmeansCluster(self, nclusters):
        """
        Clusters the data using K-means algorithm.

        Parameters:
        - nclusters: the number of clusters to form.

        Returns:
        - An array of cluster labels.
        """
        K_mod = KMeans(n_clusters=nclusters, random_state=42, n_init=20)
        K_mod.fit(self.data)
        clusters = K_mod.predict(self.data)
        return clusters

    def _build_initial_graph(self, X, topk=10):
        """
        Build an initial local graph from cosine similarity.
        This plays the role of graph-guided initialization.
        """
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 0.0)
        sim = np.maximum(sim, 0.0)

        n = sim.shape[0]
        topk = min(topk, max(1, n - 1))
        A = np.zeros_like(sim)

        for i in range(n):
            idx = np.argsort(sim[i])[-topk:]
            A[i, idx] = sim[i, idx]

        A = 0.5 * (A + A.T)
        row_sum = A.sum(axis=1, keepdims=True) + 1e-8
        A = A / row_sum
        return A

    def _self_representation_closed_form(self, X, alpha=1.0, init_graph=None):
        """
        Learn a lightweight self-representation matrix C.
        """
        n = X.shape[0]
        gram = X @ X.T
        I = np.eye(n)

        C = np.linalg.solve(gram + alpha * I, gram)
        np.fill_diagonal(C, 0.0)
        C = np.maximum(C, 0.0)

        if init_graph is not None:
            C = 0.7 * C + 0.3 * init_graph
            np.fill_diagonal(C, 0.0)

        row_sum = C.sum(axis=1, keepdims=True) + 1e-8
        C = C / row_sum
        return C

    def _good_neighbor_refine(self, C, eta=8, gamma=5, eps=1):
        """
        Refine self-representation using the good-neighbor idea.
        """
        n = C.shape[0]
        absC = np.abs(C)
        C_hat = np.zeros_like(absC)

        eta = min(eta, max(1, n - 1))
        gamma = min(gamma, eta)

        top_eta = np.argsort(absC, axis=1)[:, -eta:]

        for i in range(n):
            candidates = top_eta[i]
            scored_neighbors = []

            for j in candidates:
                neigh_j = set(top_eta[j].tolist())
                sij = 1 if i in neigh_j else 0

                if sij >= eps:
                    scored_neighbors.append((j, absC[i, j]))

            if len(scored_neighbors) < gamma:
                scored_neighbors = [(j, absC[i, j]) for j in candidates]

            scored_neighbors = sorted(scored_neighbors, key=lambda x: x[1], reverse=True)[:gamma]
            keep_idx = [j for j, _ in scored_neighbors]

            if len(keep_idx) > 0:
                vals = absC[i, keep_idx]
                vals_sum = vals.sum() + 1e-8
                C_hat[i, keep_idx] = vals / vals_sum

        C_hat = 0.5 * (C_hat + C_hat.T)
        np.fill_diagonal(C_hat, 0.0)
        return C_hat

    def _build_confidence_matrix(self, C_hat, sigma=0.6):
        """
        Build confidence matrix W from refined graph.
        """
        W = (C_hat >= sigma).astype(float)

        if W.sum() == 0:
            positive_vals = C_hat[C_hat > 0]
            if positive_vals.size > 0:
                thresh = np.percentile(positive_vals, 70)
                W = (C_hat >= thresh).astype(float)
            else:
                W = np.ones_like(C_hat)

        np.fill_diagonal(W, 0.0)
        return W

    def SelfReinforcedSubspaceCluster(
        self,
        nclusters,
        alpha=1.0,
        eta=8,
        gamma=5,
        sigma=0.6,
        init_topk=10,
        n_refine_iters=3,
        random_state=42
    ):
        """
        A lightweight self-reinforced subspace clustering module inspired by:
        Deep Self-Reinforced Multi-View Subspace Clustering for Cancer Subtyping

        Steps:
        1) Learn self-representation matrix C
        2) Refine C by good-neighbor search to obtain C_hat
        3) Build confidence matrix W
        4) Iteratively update consensus affinity S
        5) Apply spectral clustering on S
        """
        X = self.data.values if isinstance(self.data, pd.DataFrame) else np.asarray(self.data)
        X = StandardScaler().fit_transform(X)

        n = X.shape[0]
        if nclusters >= n:
            raise ValueError(f"nclusters={nclusters} must be smaller than sample size={n}")

        A_init = self._build_initial_graph(X, topk=init_topk)
        C = self._self_representation_closed_form(X, alpha=alpha, init_graph=A_init)
        S = A_init.copy()

        for _ in range(n_refine_iters):
            C_hat = self._good_neighbor_refine(C, eta=eta, gamma=gamma, eps=1)
            W = self._build_confidence_matrix(C_hat, sigma=sigma)

            S_new = W * C
            S_new = 0.5 * (S_new + S_new.T)

            if np.count_nonzero(S_new) == 0:
                S_new = C_hat.copy()

            S = 0.5 * S + 0.5 * S_new
            S = 0.5 * (S + S.T)
            np.fill_diagonal(S, 1.0)

            C = 0.7 * C + 0.3 * S
            np.fill_diagonal(C, 0.0)
            row_sum = C.sum(axis=1, keepdims=True) + 1e-8
            C = C / row_sum

        spectral = SpectralClustering(
            n_clusters=nclusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=random_state
        )
        clusters = spectral.fit_predict(S)
        return clusters

    def save_clusters(self, clusters, filename):
        """
        Save the clustered data with labels to a file.

        Parameters:
        - clusters: the cluster labels to save.
        - filename: the name of the output file.
        """
        self.sur_data['Cluster'] = clusters
        self.sur_data.to_csv(filename, index=False)

    def LogRankp(self, nclusters):
        """
        Performs the Log-rank test for clustering quality evaluation.

        Parameters:
        - nclusters: the number of clusters to evaluate.

        Returns:
        - The p-value of the Log-rank test and an array of cluster labels.
        """
        if nclusters in self.logrank_cache:
            return self.logrank_cache[nclusters]

        clusters = self.KmeansCluster(nclusters)
        self.sur_data['Type'] = clusters
        pvalue = multivariate_logrank_test(
            self.sur_data['days'],
            self.sur_data['Type'],
            self.sur_data['status']
        )

        self.logrank_cache[nclusters] = (pvalue, clusters)
        return pvalue, clusters

    def compute_indexes(self, maxclusters):
        """
        Computes and prints the clustering evaluation indexes for different cluster numbers.

        Parameters:
        - maxclusters: the maximum number of clusters to evaluate.
        """
        for i in range(2, maxclusters + 1):
            pvalue, clusters = self.LogRankp(i)
            estimator = KMeans(n_clusters=i, random_state=42, n_init=20)
            estimator.fit(self.data)

            labels = estimator.labels_

            silhouette = silhouette_score(self.data, labels, metric='cosine')
            ch_score = calinski_harabasz_score(self.data, labels)
            db_score = davies_bouldin_score(self.data, labels)
            print(f"Number of clusters: {i}")
            print(f"Silhouette Score: {silhouette}")
            print(f"Calinski-Harabasz Index: {ch_score}")
            print(f"Davies-Bouldin Index: {db_score}")
            print("P-value: ", pvalue.p_value)


def do_km_plot(survive_data, pvalue, cindex, cancer_type, model_name):
    # import necessary packages
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # extract relevant data
    values = np.asarray(survive_data['Type'])
    events = np.asarray(survive_data['status'])
    times = np.asarray(survive_data['days'])
    df = pd.DataFrame({
        'Type': values,
        'status': events,
        'days': times
    })

    # Save the dataframe to a CSV file
    output_filename = f"/public/home/tanjingbo/CA-CAE-main/CACAE/Result_929/{cancer_type}_xxx.csv"
    df.to_csv(output_filename, index=False)

    # set plotting style
    sns.set(style='ticks', context='notebook', font_scale=1.5)

    # create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # customize plot style
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # fit and plot Kaplan-Meier survival curves for each cluster
    kaplan = KaplanMeierFitter()
    for label in set(values):
        kaplan.fit(
            times[values == label],
            event_observed=events[values == label],
            label='cluster {0}'.format(label)
        )
        kaplan.plot_survival_function(ax=ax, ci_alpha=0)
        ax.legend(loc=1, frameon=False)

    # customize plot labels and title based on whether C-index was calculated or not
    if cindex is None:
        ax.set_xlabel('days', fontsize=20)
        ax.set_ylabel('Survival Probability', fontsize=20)
        ax.set_title(
            '{1} \n Cancer: {0}    p-value.{2: .1e} '.format(
                cancer_type, model_name, pvalue
            ),
            fontsize=18,
            fontweight='bold'
        )
    else:
        ax.set_xlabel('days', fontsize=20)
        ax.set_ylabel('Survival Probability', fontsize=20)
        ax.set_title(
            '{1} \n Cancer: {0}  p-value: {2: .1e}   Cindex: {3: .2f}'.format(
                cancer_type, model_name, pvalue, cindex
            ),
            fontsize=18,
            fontweight='bold'
        )
