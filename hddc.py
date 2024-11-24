import numpy as np
from scipy import linalg
from scipy.linalg.blas import dsyrk
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array
from sklearn.utils import check_random_state
from sklearn.metrics import normalized_mutual_info_score as nmi
from typing import Optional, Sequence, Union, Dict
from dataset import flat_covariances_gaussian_mixture_dataset

EPS = 1e-5

def cattell_scree_test(eigvals: Union[Sequence[float], np.ndarray], t: float = 0.2) -> int:
    """
    Perform Cattell's Scree Test to determine the optimal number of factors/components
    based on the eigenvalues of a covariance or correlation matrix, ideal for 
    applications such as Principal Component Analysis or High Dimensional Data Clustering (HDCC).

    Parameters
    ----------
    eigvals : Union[Sequence[float], np.ndarray]
        A sequence or array of eigenvalues sorted in descending order.
    t : float, optional
        Threshold ratio for identifying stabilization in eigenvalue differences, 
        by default 0.2. The parameter controls the sensitivity of the test to 
        changes in eigenvalue differences:
        - `t = 0.2` (Default): Retains clusters only after significant stabilization 
          in eigenvalue differences. Suitable for compact cluster structures with 
          strong separation.
        - `t = 0.5`: Balances sensitivity and specificity, potentially identifying 
          more clusters. Appropriate for moderately overlapping clusters.
        - `t = 0.8`: Highly sensitive to small changes, often retaining more clusters. 
          Useful for exploratory clustering where finer granularity is needed.

    Returns
    -------
    int
        The optimal number of components (index at which the scree "elbow" occurs).

    Raises
    ------
    ValueError
        If the eigenvalues are not sorted in descending order.

    Notes
    -----
    In the context of PCA. eigenvalues represent the variances explained by the
    chosen dimensions. Thus it plays the role of the right cutoff.

    In the context of HDCC, eigenvalues often represent the variances explained by
    latent factors within clusters. Cattell's Scree Test can help determine the 
    intrinsic dimensionality of cluster subspaces, a crucial step for accurately 
    modeling high-dimensional data.

    References
    ----------
    - Cattell, R. B. (1966). The Scree Test For The Number Of Factors. 
      Multivariate Behavioral Research, 1(2), 245-276. DOI:10.1207/s15327906mbr0102_10
    - Bouveyron, C., Celeux, G., Murphy, T. B., & Raftery, A. E. (2007). 
      Model-based clustering of high-dimensional data. 
      Computational Statistics & Data Analysis, 52(8), 502-519.
    """
    # Convert the input eigenvalues to a NumPy array for consistent operations.
    eigvals = np.asarray(eigvals, dtype=float)

    # Ensure eigenvalues are sorted in descending order.
    if not np.all(eigvals[:-1] >= eigvals[1:]):
        raise ValueError("Eigenvalues should be sorted in descending order.")

    # Calculate differences between consecutive eigenvalues to detect drops.
    differences = np.abs(np.diff(eigvals))

    # Find the index of the maximum difference, which is the likely "elbow" start.
    max_diff_idx = np.argmax(differences)

    # Check stabilization by traversing differences after the maximum drop.
    for i in range(max_diff_idx, len(eigvals) - 1):
        if differences[i] < t * np.max(differences):
            return i

    # If no stabilization is found, return the total number of eigenvalues.
    return len(eigvals)

class HDDC(BaseEstimator, ClusterMixin):
    """
    High Dimensional Data Clustering (HDDC) implementation.

    HDDC is a model-based clustering method tailored for high-dimensional datasets.
    It relies on the assumption that clusters are embedded in subspaces of lower
    dimensionality, and incorporates variance modeling to handle noise effectively.

    Parameters
    ----------
    n_components : int, optional, default=4
        Number of clusters to fit.
    tol : float, optional, default=100 * EPS
        Tolerance for convergence in the EM algorithm.
    max_iter : int, optional, default=100
        Maximum number of iterations for the EM algorithm.
    n_init : int, optional, default=10
        Number of random initializations to run.
    random_state : Optional[Union[int, np.random.RandomState]], optional
        Random state for reproducibility.
    cattell_threshold : float, optional, default=0.5
        Threshold for the Cattell scree test to determine subspace dimensionality.
    common_signal_dimensionality : bool, optional, default=False
        Whether all clusters share the same signal dimensionality.
    common_noise_variance : bool, optional, default=False
        Whether all clusters share the same noise variance.
    common_signal_variance_across_clusters : bool, optional, default=False
        Whether signal variances are shared across clusters.
    isotropic_signal_variance : bool, optional, default=False
        Whether signal variances are isotropic within clusters.
    common_signal_subspace_basis : bool, optional, default=False
        Whether clusters share the same subspace basis.
    min_size_cluster : int, optional, default=5
        Minimum size allowed for clusters.
    init_params : Union[str, np.ndarray], optional, default="random"
        Initialization method ('random', 'kmeans', or a custom array of labels).

    Attributes
    ----------
    responsibilities_ : np.ndarray
        Posterior probabilities for each data point belonging to each cluster.
    weights_ : np.ndarray
        Weights (proportions) of each cluster.
    means_ : np.ndarray
        Cluster centroids.
    eigenvalues_ : List[np.ndarray]
        Eigenvalues for each cluster's covariance matrix.
    eigenvectors_ : List[np.ndarray]
        Eigenvectors for each cluster's covariance matrix.
    signal_dims_ : List[int]
        Dimensionality of the signal subspace for each cluster.
    noise_variances_ : np.ndarray
        Noise variances for each cluster.
    log_likelihood_ : float
        Log-likelihood of the current model.

    Notes
    -----
    The HDDC model is based on Gaussian Mixture Models (GMMs), with specific
    constraints for high-dimensional data (e.g. rank constraints on covariances).
    It uses subspace projections and variance modeling to cluster effectively
    in high-dimensional spaces.

    References
    ----------
    - Bouveyron, C., Celeux, G., Murphy, T. B., & Raftery, A. E. (2007).
    Model-based clustering of high-dimensional data.
    Computational Statistics & Data Analysis, 52(8), 502-519.
    """

    def __init__(
        self,
        n_components: int = 4,
        tol: float = 100.0 * EPS,
        max_iter: int = 100,
        n_init: int = 10,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        cattell_threshold: float = 0.5,
        common_signal_dimensionality: bool = False,
        common_noise_variance: bool = False,
        common_signal_variance_across_clusters: bool = False,
        isotropic_signal_variance: bool = False,
        common_signal_subspace_basis: bool = False,
        min_size_cluster: int = 5,
        init_params: Union[str, np.ndarray] = "random",
        verbose: bool = False
    ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cattell_threshold = cattell_threshold
        self.common_signal_dimensionality = common_signal_dimensionality
        self.common_noise_variance = common_noise_variance
        self.common_signal_variance_across_clusters = common_signal_variance_across_clusters
        self.isotropic_signal_variance = isotropic_signal_variance
        self.common_signal_subspace_basis = common_signal_subspace_basis
        self.min_size_cluster = min_size_cluster
        self.init_params = init_params
        self.verbose = verbose

        if not isinstance(init_params, str):
            n_init = 1
            self.n_init = n_init


    def _n_parameters(self) -> int:
        """
        Compute the total number of free parameters in the HDDC model.

        This method calculates the number of parameters used in the model, 
        considering constraints like shared signal/noise variances or dimensionalities. 
        The parameter count is crucial for model selection criteria such as AIC and BIC.

        Returns
        -------
        int
            Total number of free parameters in the model.
        """
        # 1. Mixing proportions: (n_components - 1) parameters as they sum to 1.
        params = self.n_components - 1

        # 2. Cluster means: Each cluster has n_features mean values.
        params += self.n_components * self.n_features_

        # 3. Signal subspace parameters for each cluster:
        # For each cluster k:
        # - d_k * (n_features - d_k / 2):
        #   * d_k: dimensionality of the signal subspace (number of eigenvectors retained).
        #   * (n_features - d_k / 2): degrees of freedom for signal variance and subspace.
        for k in range(self.n_components):
            d_k = self.signal_dims_[k]
            params += d_k * (self.n_features_ - d_k / 2)

        # 4. Adjustments for shared constraints:
        # Shared signal dimensionality: Reduces the number of free parameters for dimensionality.
        if self.common_signal_dimensionality:
            params -= (self.n_components - 1)  # Shared signal dimensionality has only 1 parameter.

        # Shared noise variance: Reduces the number of free noise variance parameters.
        if self.common_noise_variance:
            params -= (self.n_components - 1)  # Shared noise variance has only 1 parameter.

        # Shared signal variance across clusters:
        # If signal variance is shared, each cluster loses its variance parameter.
        if self.common_signal_variance_across_clusters:
            params -= self.n_components

        # Isotropic signal variance: Each cluster loses d_k - 1 parameters (kept 1 for isotropic variance).
        if self.isotropic_signal_variance:
            params -= self.n_components

        return int(params)


    def _initialize(self, X: np.ndarray) -> None:
        self.random_state_ = check_random_state(self.random_state)
        self.n_samples_, self.n_features_ = X.shape

        # Initialization of responsibilities
        if isinstance(self.init_params, np.ndarray):
            if self.init_params.shape[0] != self.n_samples_:
                raise ValueError("`init_params` array must have the same length as the number of samples.")
            if not np.all(np.isin(self.init_params, range(self.n_components))):
                raise ValueError("`init_params` array contains invalid cluster indices.")
            labels = self.init_params
            self.responsibilities_ = np.eye(self.n_components)[labels]
        elif self.init_params == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state_)
            labels = kmeans.fit_predict(X)
            self.responsibilities_ = np.eye(self.n_components)[labels]
        elif self.init_params == 'random':
            self.responsibilities_ = self.random_state_.rand(self.n_samples_, self.n_components)
            self.responsibilities_ /= self.responsibilities_.sum(axis=1, keepdims=True)
        else:
            raise ValueError("`init` must be one of {'kmeans', 'random'} or overridden by `labels`.")

        self.weights_ = self.responsibilities_.sum(axis=0) / self.n_samples_
        self.means_ = np.dot(self.responsibilities_.T, X) / self.weights_[:, np.newaxis]
        self.eigenvalues_ = []
        self.eigenvectors_ = []
        self.signal_dims_ = []
        self.noise_variances_ = np.zeros(self.n_components)

        for k in range(self.n_components):
            cluster_data = X - self.means_[k]
            cov_matrix = np.dot(cluster_data.T * self.responsibilities_[:, k], cluster_data) / self.responsibilities_[:, k].sum()

            eigvals, eigvecs = linalg.eigh(cov_matrix)
            eigvals = np.maximum(eigvals[::-1], EPS)  # Sort descending and ensure numerical stability
            eigvecs = eigvecs[:, ::-1]  # Align eigenvectors with sorted eigenvalues

            self.eigenvalues_.append(eigvals)
            self.eigenvectors_.append(eigvecs)
            d = cattell_scree_test(eigvals, t=self.cattell_threshold)
            self.signal_dims_.append(d)

        # Apply common signal dimensionality
        if self.common_signal_dimensionality:
            self.shared_signal_dim_ = max(self.signal_dims_)
            self.signal_dims_ = [self.shared_signal_dim_] * self.n_components

        # Apply common noise variance
        if self.common_noise_variance:
            all_noise_vars = [np.mean(eigvals[d:]) for eigvals, d in zip(self.eigenvalues_, self.signal_dims_)]
            self.shared_noise_variance_ = np.mean(all_noise_vars)
            self.noise_variances_ = [self.shared_noise_variance_] * self.n_components

        # Apply common signal variance across clusters
        if self.common_signal_variance_across_clusters:
            for k in range(self.n_components):
                self.eigenvalues_[k][:self.signal_dims_[k]] = np.mean(
                    [self.eigenvalues_[j][:self.signal_dims_[j]] for j in range(self.n_components)], axis=0
                )

        # Apply isotropic signal variance
        if self.isotropic_signal_variance:
            for k in range(self.n_components):
                signal_variance = np.mean(self.eigenvalues_[k][:self.signal_dims_[k]])
                self.eigenvalues_[k][:self.signal_dims_[k]] = signal_variance

        # Apply common signal subspace basis
        if self.common_signal_subspace_basis:
            mean_signal_basis = np.mean(
                [self.eigenvectors_[k][:, :self.signal_dims_[k]] for k in range(self.n_components)],
                axis=0,
            )
            for k in range(self.n_components):
                self.eigenvectors_[k][:, :self.signal_dims_[k]] = mean_signal_basis

    def _m_step(self, X: np.ndarray) -> None:
        """
        Perform the M-step: update model parameters based on current responsibilities.

        The M-step updates the model parameters, including cluster weights, means,
        covariance structure, and other constraints, using the expected cluster 
        assignments (responsibilities) computed in the E-step.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
        """
        # Compute the effective number of data points assigned to each cluster (N_k)
        N_k = self.responsibilities_.sum(axis=0)

        # Ensure clusters meet the minimum size requirement
        while np.min(N_k) < self.min_size_cluster:
            k_min = np.argmin(N_k)  # Find the cluster with the smallest size
            p = self.responsibilities_.sum(axis=1)  # Compute total responsibilities for each point
            p = np.clip(1 - p, EPS, None)  # Avoid zero probabilities
            p /= p.sum()  # Normalize to create a valid probability distribution
            missing_points = round(self.min_size_cluster - N_k[k_min])  # Points needed to meet the minimum size
            idx = self.random_state_.choice(len(p), size=missing_points, replace=False, p=p)  # Select points
            self.responsibilities_[idx, k_min] = 1  # Assign these points to the smallest cluster
            N_k = self.responsibilities_.sum(axis=0)  # Recompute cluster sizes

        # Update cluster weights (proportions)
        self.weights_ = N_k / self.n_samples_

        # Update cluster means using responsibilities as weights
        self.means_ = np.dot(self.responsibilities_.T, X) / N_k[:, np.newaxis]

        # Variables to handle constraints on signal/noise properties
        signal_variances = []  # Track signal variances for isotropic or shared constraints

        # Update covariance-related parameters for each cluster
        for k in range(self.n_components):
            # Compute weighted deviations from the mean
            resp_sqrt = np.sqrt(self.responsibilities_[:, k])[:, np.newaxis]  # Weighted square root of responsibilities
            centered = (X - self.means_[k]) * resp_sqrt
            cov_matrix = np.dot(centered.T, centered) / N_k[k]  # Compute covariance matrix
            cov_matrix = 0.5 * (cov_matrix + cov_matrix.T) + EPS * np.eye(self.n_features_)  # Ensure symmetry and stability

            # Perform eigen decomposition of the covariance matrix
            eigvals, eigvecs = linalg.eigh(cov_matrix)
            eigvals = eigvals[::-1]  # Sort eigenvalues in descending order
            eigvecs = eigvecs[:, ::-1]  # Align eigenvectors with sorted eigenvalues

            # Store updated eigenvalues and eigenvectors
            self.eigenvalues_[k] = eigvals
            self.eigenvectors_[k] = eigvecs

            # Determine signal dimensionality using the Cattell scree test
            if not self.common_signal_dimensionality:
                self.signal_dims_[k] = cattell_scree_test(eigvals, t=self.cattell_threshold)

            # Calculate noise variance for the cluster
            if not self.common_noise_variance:
                self.noise_variances_[k] = max(np.mean(eigvals[self.signal_dims_[k]:]), EPS)

            # Track signal variances for further constraints
            signal_variances.append(np.mean(eigvals[:self.signal_dims_[k]]))

        # Apply shared constraints on parameters if specified
        if self.common_signal_dimensionality:
            # Use the maximum signal dimensionality across clusters
            shared_signal_dim = max(self.signal_dims_)
            self.signal_dims_ = [shared_signal_dim] * self.n_components

        if self.common_noise_variance:
            # Compute and use shared noise variance across clusters
            shared_noise_variance = np.mean(
                [np.mean(self.eigenvalues_[k][self.signal_dims_[k]:]) for k in range(self.n_components)]
            )
            self.noise_variances_ = [shared_noise_variance] * self.n_components

        if self.common_signal_variance_across_clusters:
            # Compute and enforce shared signal variance across clusters
            shared_signal_variance = np.mean(signal_variances)
            for k in range(self.n_components):
                self.eigenvalues_[k][:self.signal_dims_[k]] = shared_signal_variance

        if self.isotropic_signal_variance:
            # Enforce isotropic signal variance within clusters
            for k in range(self.n_components):
                isotropic_variance = np.mean(self.eigenvalues_[k][:self.signal_dims_[k]])
                self.eigenvalues_[k][:self.signal_dims_[k]] = isotropic_variance

        if self.common_signal_subspace_basis:
            # Compute and enforce a shared signal subspace basis
            shared_signal_basis = np.mean(
                [self.eigenvectors_[k][:, :self.signal_dims_[k]] for k in range(self.n_components)],
                axis=0,
            )
            for k in range(self.n_components):
                self.eigenvectors_[k][:, :self.signal_dims_[k]] = shared_signal_basis


    def _compute_log_density(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        Compute the log-density for a given cluster `k` and dataset `X`.

        Parameters
        ----------
        X : np.ndarray
            The data matrix of shape (n_samples, n_features).
        k : int
            The index of the cluster for which the log-density is calculated.

        Returns
        -------
        log_density : np.ndarray
            The log-density for each data point in `X` for cluster `k`.
        """
        # Center the data relative to the cluster mean
        diff = X - self.means_[k]

        # Retrieve the eigenvalues (signal and noise) and eigenvectors for the cluster
        eigenvalues = np.maximum(self.eigenvalues_[k], EPS)  # Ensure numerical stability
        d = self.signal_dims_[k]  # Signal dimensionality
        noise_variance = np.maximum(self.noise_variances_[k], EPS)  # Avoid very small variances

        # Compute the log determinant of the covariance matrix
        # The log determinant accounts for the signal and noise contributions
        log_det = (
            np.sum(np.log(eigenvalues[:d]))  # Log of signal eigenvalues
            + (self.n_features_ - d) * np.log(noise_variance)  # Log of noise variance for residuals
        )

        # Project the data onto the signal subspace
        # Compute the contribution of the signal subspace to the Mahalanobis distance
        signal_proj = np.dot(diff, self.eigenvectors_[k][:, :d])  # Project onto top d eigenvectors
        signal_contrib = np.sum((signal_proj**2) / eigenvalues[:d], axis=1)  # Weighted by eigenvalues

        # Compute the residual (noise) contribution
        # Residual is the difference between the original data and its projection onto the signal subspace
        residual = diff - np.dot(signal_proj, self.eigenvectors_[k][:, :d].T)  # Remove signal contribution
        noise_contrib = np.sum(residual**2, axis=1) / noise_variance  # Scale residuals by noise variance

        # Combine all terms to compute the log-density
        # The log-density is composed of the log determinant and the Mahalanobis distance contributions
        log_density = (
            -0.5 * log_det  # Contribution of the log determinant
            - 0.5 * (signal_contrib + noise_contrib)  # Combined Mahalanobis distance
        )

        return log_density


    def _e_step(self, X: np.ndarray) -> float:
        """
        Perform the E-step: calculate the responsibilities based on the current model parameters.

        The E-step computes the posterior probabilities (responsibilities) for each data 
        point belonging to each cluster, using the current estimates of the model parameters.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        float
            Log-likelihood of the data under the current model parameters.
        """
        # Initialize the log responsibilities matrix
        # Shape: (n_samples, n_components)
        log_resp = np.zeros((self.n_samples_, self.n_components))

        # Compute log-densities for each cluster
        for k in range(self.n_components):
            # log_resp[:, k] stores the log-density of each sample for cluster k
            log_resp[:, k] = self._compute_log_density(X, k)

        # Combine log-densities with cluster weights (log(pi_k))
        # log(pi_k) is the logarithm of the cluster weight for cluster k
        log_resp += np.log(self.weights_)

        # Normalize responsibilities in the log domain to avoid underflow
        # Subtract the log-sum-exp for numerical stability
        log_resp = log_resp - logsumexp(log_resp, axis=1, keepdims=True)

        # Convert log responsibilities to probabilities (exp for numerical stability)
        self.responsibilities_ = np.exp(log_resp)

        # Calculate and return the log-likelihood
        # The log-likelihood is the sum of the log-sum-exp values across all samples
        log_likelihood = logsumexp(log_resp, axis=1).sum()

        return log_likelihood

    def _run_em(self, X: np.ndarray) -> None:
        """
        Run the Expectation-Maximization (EM) algorithm.

        This method alternates between the E-step and the M-step until the model converges
        or the maximum number of iterations is reached. Convergence is determined by the 
        change in log-likelihood falling below a specified tolerance.

        Parameters
        ----------
        X : np.ndarray
            The data matrix of shape (n_samples, n_features).
        """
        # Initialize convergence tracking variables
        prev_log_likelihood = -np.inf  # Set previous log-likelihood to a very low value
        self.log_likelihood_ = -np.inf  # Initialize the current log-likelihood

        for iteration in range(self.max_iter):
            # Perform the E-step: update responsibilities and compute log-likelihood
            self.log_likelihood_ = self._e_step(X)
            if self.verbose:
                s = "%05d" % iteration
                ell = self.log_likelihood_ 
                print(f"Iteration:\t{s}\t\tLikelihood:\t{ell}")

            # Perform the M-step: update model parameters based on responsibilities
            self._m_step(X)

            # Check for convergence: if the log-likelihood change is below tolerance, stop
            if abs(self.log_likelihood_ - prev_log_likelihood) < self.tol:
                if self.verbose:
                    print("Stopped because log likelihood ceased to evolve")
                break

            # Update previous log-likelihood for the next iteration
            prev_log_likelihood = self.log_likelihood_


    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "HDDC":
        """
        Fit the HDDC model to the data.

        This method performs the Expectation-Maximization (EM) algorithm over multiple
        random initializations to optimize the model parameters. The best parameters
        are retained based on the highest log-likelihood achieved.

        Parameters
        ----------
        X : np.ndarray
            The data matrix of shape (n_samples, n_features).
        y : Optional[np.ndarray], default=None
            Ignored. Added for API compatibility with scikit-learn.

        Returns
        -------
        self : HDDC
            The fitted HDDC instance.
        """
        # Validate the input data
        X = check_array(X, dtype=np.float64, ensure_min_samples=2)
        self.n_samples_, self.n_features_ = X.shape  # Store data dimensions

        # Track the best log-likelihood and corresponding parameters
        best_log_likelihood = -np.inf
        best_params = None

        # Iterate over multiple initializations
        for init_run in range(self.n_init):
            # Step 1: Initialize the model parameters and responsibilities
            self._initialize(X)

            # Step 2: Run the EM algorithm
            self._run_em(X)

            # Step 3: Check if the current initialization achieves a better log-likelihood
            if self.log_likelihood_ > best_log_likelihood:
                # Update the best log-likelihood
                best_log_likelihood = self.log_likelihood_

                # Save the current parameters as the best ones
                best_params = {
                    "weights_": self.weights_,
                    "means_": self.means_,
                    "eigenvalues_": self.eigenvalues_,
                    "eigenvectors_": self.eigenvectors_,
                    "signal_dims_": self.signal_dims_,
                    "noise_variances_": self.noise_variances_,
                    "responsibilities_": self.responsibilities_,
                }

        # Restore the best parameters after all initializations
        if best_params is not None:
            self.weights_ = best_params["weights_"]
            self.means_ = best_params["means_"]
            self.eigenvalues_ = best_params["eigenvalues_"]
            self.eigenvectors_ = best_params["eigenvectors_"]
            self.signal_dims_ = best_params["signal_dims_"]
            self.noise_variances_ = best_params["noise_variances_"]
            self.responsibilities_ = best_params["responsibilities_"]
            self.log_likelihood_ = best_log_likelihood

        # Predict cluster labels based on the best responsibilities
        self.labels_ = self.predict(X)

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each data point in `X` to a cluster.

        This method computes the posterior probabilities (responsibilities) for each
        cluster and assigns each sample to the cluster with the highest responsibility.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        labels : np.ndarray
            Cluster assignments for each sample, with shape (n_samples,).
        """
        # Validate the input data
        X = check_array(X, dtype=np.float64)

        # Initialize log responsibilities matrix
        # Shape: (n_samples, n_components)
        log_resp = np.zeros((X.shape[0], self.n_components))

        # Compute log-densities for each cluster
        for k in range(self.n_components):
            # log_resp[:, k] stores the log-density of each sample for cluster k
            log_resp[:, k] = self._compute_log_density(X, k)

        # Normalize log responsibilities
        log_resp = log_resp - logsumexp(log_resp, axis=1, keepdims=True)

        # Convert log responsibilities to probabilities
        responsibilities = np.exp(log_resp)

        # Assign each sample to the cluster with the highest responsibility
        labels = responsibilities.argmax(axis=1)

        return labels


    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the HDDC model to the data and predict cluster labels.

        This method combines the functionality of `fit` and `predict` by first fitting
        the model to the data and then assigning each sample to a cluster.

        Parameters
        ----------
        X : np.ndarray
            The data matrix of shape (n_samples, n_features).
        y : Optional[np.ndarray], default=None
            Ignored. Included for API compatibility with scikit-learn.

        Returns
        -------
        labels : np.ndarray
            Cluster assignments for each sample, with shape (n_samples,).
        """
        # Fit the HDDC model to the data
        self.fit(X, y)

        # Predict cluster labels for the data
        labels = self.predict(X)

        return labels

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the log-probability (log-density) of each sample under the model.

        This method calculates the log-probability of the data points under the fitted
        HDDC model. The log-density is useful for tasks such as anomaly detection or
        evaluating the fit of the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        log_probs : np.ndarray
            Log-probability of each sample, with shape (n_samples,).
        """
        # Validate the input data
        X = check_array(X, dtype=np.float64)

        # Initialize log responsibilities matrix
        # Shape: (n_samples, n_components)
        log_resp = np.zeros((X.shape[0], self.n_components))

        # Compute log-densities for each cluster
        for k in range(self.n_components):
            # log_resp[:, k] stores the log-density of each sample for cluster k
            log_resp[:, k] = self._compute_log_density(X, k)

        # Add log of cluster weights (log(pi_k))
        log_resp += np.log(self.weights_)

        # Compute the log-sum-exp across clusters for each sample
        # This gives the log-probability of each sample under the mixture model
        log_probs = logsumexp(log_resp, axis=1)

        return log_probs


    def score(self, X: np.ndarray) -> float:
        """
        Compute the average log-likelihood of the data under the model.

        The score method calculates the total log-probability of the data and 
        normalizes it by the number of samples. This serves as a measure of the 
        overall goodness-of-fit of the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        average_log_likelihood : float
            The average log-likelihood of the data.
        """
        # Validate the input data
        X = check_array(X, dtype=np.float64)

        # Compute the total log-probabilities for all samples
        total_log_likelihood = self.score_samples(X).sum()

        # Compute and return the average log-likelihood
        average_log_likelihood = total_log_likelihood / X.shape[0]
        return average_log_likelihood


    def aic(self, X: np.ndarray) -> float:
        """
        Compute the Akaike Information Criterion (AIC) for the HDDC model.

        The AIC is a measure of model quality that balances goodness-of-fit and model complexity.
        It is defined as:
            AIC = -2 * log-likelihood + 2 * number of parameters
        Lower values of AIC indicate a better balance between model fit and complexity.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        aic_value : float
            The AIC value for the HDDC model (lower is better).
        """
        # Validate the input data
        X = check_array(X, dtype=np.float64)

        # Compute the total log-likelihood of the data
        total_log_likelihood = self.score_samples(X).sum()

        # Compute the number of free parameters in the model
        n_parameters = self._n_parameters()

        # Calculate AIC using the formula: AIC = -2 * log-likelihood + 2 * n_parameters
        aic_value = -2 * total_log_likelihood + 2 * n_parameters

        return aic_value


    def bic(self, X: np.ndarray) -> float:
        """
        Compute the Bayesian Information Criterion (BIC) for the HDDC model.

        The BIC evaluates model quality by balancing goodness-of-fit and complexity,
        with a stronger penalty for model complexity compared to AIC. It is defined as:
            BIC = -2 * log-likelihood + number of parameters * log(number of samples)
        Lower values of BIC indicate a better trade-off between model fit and complexity.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        bic_value : float
            The BIC value for the HDDC model (lower is better).
        """
        # Validate the input data
        X = check_array(X, dtype=np.float64)

        # Compute the total log-likelihood of the data
        total_log_likelihood = self.score_samples(X).sum()

        # Compute the number of free parameters in the model
        n_parameters = self._n_parameters()

        # Number of samples in the dataset
        n_samples = X.shape[0]

        # Calculate BIC using the formula: BIC = -2 * log-likelihood + n_parameters * log(n_samples)
        bic_value = -2 * total_log_likelihood + n_parameters * np.log(n_samples)

        return bic_value


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the posterior probabilities (responsibilities) for each cluster.

        This method computes the probability of each sample belonging to each cluster 
        based on the fitted HDDC model. These probabilities represent the strength of 
        assignment to each cluster.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        responsibilities : np.ndarray
            Posterior probabilities for each sample and cluster, with shape 
            (n_samples, n_components).
        """
        # Validate the input data
        X = check_array(X, dtype=np.float64)

        # Initialize the log responsibilities matrix
        # Shape: (n_samples, n_components)
        log_resp = np.zeros((X.shape[0], self.n_components))

        # Compute log-densities for each cluster
        for k in range(self.n_components):
            # log_resp[:, k] stores the log-density of each sample for cluster k
            log_resp[:, k] = self._compute_log_density(X, k)

        # Add log(cluster weights) to log-densities
        log_resp += np.log(self.weights_)

        # Normalize log responsibilities using log-sum-exp for numerical stability
        log_resp -= logsumexp(log_resp, axis=1, keepdims=True)

        # Convert log responsibilities to probabilities
        responsibilities = np.exp(log_resp)

        return responsibilities


    def icl(self, X: np.ndarray) -> float:
        """
        Compute the Integrated Completed Likelihood (ICL) criterion for the HDDC model.

        The ICL criterion is a model selection metric that penalizes the Bayesian 
        Information Criterion (BIC) with the entropy of the posterior probabilities 
        (responsibilities). It favors simpler models with better-separated clusters 
        by discouraging overlapping cluster assignments.

        ICL = BIC - Posterior Entropy

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        icl_value : float
            The ICL value for the HDDC model (higher is better).
        """
        # Validate the input data
        X = check_array(X, dtype=np.float64)

        # Compute the BIC value
        bic_value = self.bic(X)

        # Compute the posterior probabilities (responsibilities)
        responsibilities = self.predict_proba(X)

        # Avoid log(0) by clipping responsibilities
        clipped_responsibilities = np.clip(responsibilities, EPS, None)

        # Explicitly discard terms where responsibilities <= EPS
        entropy_terms = clipped_responsibilities * np.log(clipped_responsibilities)
        entropy_terms[clipped_responsibilities <= EPS] = 0  # Discard negligible contributions
        entropy = -np.sum(entropy_terms)

        # Calculate ICL as: ICL = BIC - Posterior Entropy
        icl_value = bic_value - entropy

        return icl_value

