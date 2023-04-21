[![PyPI Version](https://img.shields.io/pypi/v/dimensionality_reductions_jmsv)](https://pypi.org/project/dimensionality_reductions_jmsv/)
[![Package Status](https://img.shields.io/pypi/status/dimensionality_reductions_jmsv)](https://pypi.org/project/dimensionality_reductions_jmsv/)
![Python Versions](https://img.shields.io/pypi/pyversions/dimensionality_reductions_jmsv)
[![License](https://img.shields.io/pypi/l/dimensionality_reductions_jmsv)](https://mit-license.org/)

### What is it?

**dimensionality_reductions_jmsv** is a Python package that provides three methods (PCA, SVD, t-SNE) to apply dimensionality reduction to any dataset. Aslo provides two methods (KMeans y KMedoids) to clustering. 

### Installing the package

1. Requests is available on PyPI:
    ```bash
    pip install dimensionality_reductions_jmsv
    ```

2. Try your first **_dimensionality reduction with PCA_**
    ```python
    from dimensionality_reductions_jmsv.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt

    # Generating the sample data from np.random.rand
    np.random.seed(3)
    X = np.random.rand(100, 4)

    # Apply PCA with two principal components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Calculate the covariance matrix of X
    covariance_matrix = np.cov(X.T)
    # Calculate the eigenvalues and eigenvectors of the covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    # Calculate the proportion of variance explained
    explained_variance_ratio = sorted_eigenvalues[:2] / np.sum(sorted_eigenvalues)

    # Plot the proportion of variance explained
    fig, ax = plt.subplots()
    bars = ax.bar([1, 2], explained_variance_ratio, tick_label=['1', '2'])
    plt.xlabel('# Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.title('Proportion of Variance Explained for each Principal Component')
    plt.show()
    ```

3. Try your first **_KMeans cluster_**
   ```python
    from dimensionality_reductions_jmsv.cluster import KMeans
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    # Generating the sample data from make_blobs
    X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True,
                      random_state=1, )
     
    # Initialize the clusterer with n_clusters=4 and a random generator
    k = KMeans(n_clusters=4, random_state=42)
    m = k.fit_transform(X)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Remove axis labels on all subplots
    for ax in (ax1, ax2):
        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Plot first subplot
    ax1.scatter(X[:,0], X[:,1])
    ax1.set_title('Original Dataset')

    # Plot second subplot
    ax2.scatter(X[:,0], X[:,1], c=k._assign_clusters(X))
    ax2.set_title('With KMeansCluster')
    plt.show();
   ```

