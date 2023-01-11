                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:00:03 2021

@author: Theo.ROSSI
"""

def viridis_colors(num):
    
    '''
    
    Create a list of colors
    
    num: int
        Number of returned colors.
    
    Returns:     clr: list
                    Color names.
                    
    '''
    
    clr = []
    cmap = cm.viridis(np.linspace(0.1,0.85,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr

    


def PrinCompAn(df_pca, threshold=0.95):
    
    '''
    
    Compute Pincipal Component Analysis on a dataset returns the explained variance
    of each principal component (PC) and the eigenvalues of each individual.
        
    df_pca: DataFrame of shape (n_samples, n_variables)
    
    threshold: float, default=0.95
        Between 0 and 1. Set the number of principal components explaining 'threshold' of the total variance.
        
    Figures:
        - 1: Explained variance for each principal component and 2D scatter plot (x: PC1, y: PC2).
        - 2: 3D scatter plot (x: PC1, y: PC2, z: PC3).
        - 3: Contribution of each variable on each principal component kept.
    
    Return:     ExpVar: ndarray
                    Explained variance for each principal component.
                
                df_eigen: DataFrame of shape (n_samples, n_component)
                    Dataframe of eigenvalues for each sample on each principal component kept.
                    
    '''
    
    #Determine the amount of principal components to keep. 
    pca = PCA()
    pca_fit = pca.fit(df_pca)
    ExpVar = pca_fit.explained_variance_ratio_
    x = np.arange(1, len(ExpVar) + 1)
    cumsum = np.cumsum(ExpVar)
    set_threshold = np.argmax(cumsum >= threshold) + 1
    variable_contribution = pd.DataFrame(pca_fit.components_[:, :set_threshold])
    
    
    #Apply PCA with the amount of principal component determined by the threshold.
    PrCoAn = PCA(n_components = set_threshold)
    eigenvalue = PrCoAn.fit_transform(df_pca)
    df_eigen = pd.DataFrame(eigenvalue, columns = [f'PC{i+1}' for i in range(set_threshold)])

    #Figure1: Explained variance for each principal component and 2D scatter plot (x: PC1, y: PC2)
    fig_PCA, ax_PCA = plt.subplots(1, 3, figsize = (17,4), tight_layout = True)
    
    ax_PCA[0].plot(x, cumsum, marker='o', linestyle='--', color='b')
    ax_PCA[0].set_xlabel('Number of Principal Components')
    ax_PCA[0].set_ylabel('Cumulative variance (%)')
    ax_PCA[0].set_xticks(x)
    ax_PCA[0].set_title('Number of components\nneeded to explain variance')
    ax_PCA[0].axhline(y = threshold, color = 'r', linestyle = '--')
    ax_PCA[0].text(0.5, threshold + 0.01, f'{threshold*100}% total variance', color = 'red', fontsize = 10)
    ax_PCA[0].grid(axis = 'x')
    ax_PCA[1].bar(x, ExpVar)
    ax_PCA[1].set_title("Explained variance")
    ax_PCA[1].set_ylabel("Norm variance")
    ax_PCA[1].set_xlabel("#Component")

    [ax_PCA[2].scatter(df_eigen['PC1'][eigen], df_eigen['PC2'][eigen], color='k', alpha=0.4) for eigen in range(df_eigen.shape[0])]
    [ax_PCA[2].annotate(i, (df_eigen['PC1'][i], df_eigen['PC2'][i])) for i, txt in enumerate(range(df_eigen.shape[0]))]
    ax_PCA[2].set_title('PCA')
    ax_PCA[2].set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
    ax_PCA[2].set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
    ax_PCA[2].axvline(x=0.0,color='k',linestyle='--')
    ax_PCA[2].axhline(y=0.0,color='k',linestyle='--')
    
    
    #Figure2: 3D scatter plot (x: PC1, y: PC2, z: PC3).
    if df_eigen.shape[1] == 2:
        pass
    else:
        plt.figure()
        ax_pca3d = plt.axes(projection='3d')
        
        [ax_pca3d.scatter3D(df_eigen['PC1'][eigen], df_eigen['PC2'][eigen], df_eigen['PC3'][eigen], color='k', alpha=0.4) for eigen in range(df_eigen.shape[0])]
        [ax_pca3d.text(x, y, z, label) for x, y, z, label in zip(df_eigen['PC1'], df_eigen['PC2'], df_eigen['PC3'], range(df_eigen.shape[0]))] 
        
        ax_pca3d.set_title('PCA')
        ax_pca3d.set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
        ax_pca3d.set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
        ax_pca3d.set_zlabel('PC3 ({:.2f}%)'.format(ExpVar[2]*100))
    
    
    #Figure3: Contribution of each variable on each principal component kept.
    fig_contribution, ax_contribution = plt.subplots(1, set_threshold, figsize=(17,4), sharey = True, tight_layout = True)
    
    for plot in range(set_threshold):
        ax_contribution[plot].set_title(f'PC{plot+1} contributions')
        ax_contribution[plot].bar(x, variable_contribution[plot])
        ax_contribution[plot].set_xticks(x)
        ax_contribution[plot].set_xticklabels(df_pca.columns, rotation='vertical', fontsize=7)
        ax_contribution[plot].set_ylabel('Variable contribution')
        ax_contribution[plot].axhline(y=0.0, color='k', linestyle='--')
    
    
    #Figure4: Polar plot for contribution of each variable.
    figure, correlation_matrix = plot_pca_correlation_graph(df_pca, df_pca.columns, dimensions=(1,2), figure_axis_size=5)

    return ExpVar, df_eigen




def Tsne(df_tSNE, perplexity, iteration, classes = 1, num_fig = 5):
    
    '''
    
    Compute t-SNE dimensional reduction on a dataset.
        /!\ FOR VISUALIZATION ONLY
    
    df_tSNE: DataFrame of shape (n_samples, n_variables)
    
    perplexity: int or float
        Number of nearest neighbors. Consider selecting a value between 5 and 50.
    
    iteration: int
        Maximum number of iterations for the optimization.
    
    classes: int, ndarray or list, default=None
        Number of different classes (labels) in the dataset. If int, all dots have the same color.
    
    num_fig: int, default=5
        Number of figures to generate.
    
    Figures:
        The amount of figures is determined by num_fig. Each figure shows a particular
        representation of the dataset in the embedded space.
        
    '''

    labels = range(df_tSNE.shape[0])
    
    for i in range(num_fig):
        tsne = TSNE(perplexity=perplexity, n_iter=iteration).fit_transform(df_tSNE)
        df_tsne = pd.concat((pd.DataFrame(labels, columns=['Labels']),
                            pd.DataFrame(tsne, columns=['Dim1','Dim2'])), axis=1)

        fig, ax = plt.subplots(figsize=(4,3))
        
        if type(classes) == int:
            [ax.scatter(df_tsne['Dim1'][j], df_tsne['Dim2'][j], color='#005172', alpha=0.5) for j, txt in enumerate(range(df_tsne.shape[0]))]
            
        else:
            clr = viridis_colors(np.max(classes)+1)
            [ax.scatter(df_tsne['Dim1'][j], df_tsne['Dim2'][j], color=clr[classes[j]], alpha=0.5) for j, txt in enumerate(range(df_tsne.shape[0]))]
            
        [ax.annotate(j,(df_tsne['Dim1'][j], df_tsne['Dim2'][j]), color='k') for j, txt in enumerate(range(df_tsne.shape[0]))]
                
        ax.set_title('Perplexity: {}'.format(perplexity))




def outliers_detection(df_out, outliers_proportion = 'auto', outliers_off = False):
    
    '''
    
    Detects the outliers in the dataset using the isolation forest algorithm.
    
    df_out: DataFrame of shape (n_samples, n_variables)
    
    outliers_proportion: float, default='auto'
        Proportion of outliers in the dataset.
            - auto: determines automatically the amount of outliers.
            - float: set arbitrarily the amount of outliers.
    
    outliers_off: bool, default=False
        If True, drops the outliers from the original dataset and put them in a new DataFrame.
    
    Figures:
        2D scatter plot of the dataset with outliers in red. If df_out.columns[2] != 'Labels_',
        creats a 3D scatter plot.
    
    Returns:    df_outliers: DataFrame of shape (n_samples, n_variables)
                    Outliers determined by the algorithm.
                
                df: Dataframe of shape (n_samples, n_variables)
                    If outliers_off=True, df is the original dataset without the outliers. 
        
    '''
    
    model_isolation = IsolationForest(contamination = outliers_proportion)
    model_isolation.fit(df_out)
    outliers_ = model_isolation.predict(df_out) == -1

    df_outliers = df_out[outliers_]
    print('------------------')
    print('Outliers')
    print('------------------')
    print(df_outliers)
    print('------------------')

    if outliers_off == True:
        
        df = pd.concat([df_out, pd.DataFrame(df_out.index, columns = ['Labels_'])], axis=1)
        df = df.drop(df_outliers.index, axis = 0)
        print('------------------')
        print('Dataset without outliers')
        print('------------------')
    
    else:
        df = df_out
        print('------------------')
        print('Dataset')
        print('------------------')
    
    print(df)
    print('------------------')
    
    
    #2D scatter plot of the dataset with outliers in red.
    plt.figure()
    plt.scatter(df_out.iloc[:,0], df_out.iloc[:,1], color = 'k', alpha = 0.4)
    plt.scatter(df_out.iloc[:,0][outliers_], df_out.iloc[:,1][outliers_], color = 'r')
    [plt.annotate(i, (df_out.iloc[i,0], df_out.iloc[i,1])) for i, txt in enumerate(range(df_out.shape[0]))] 
    plt.xlabel('{}'.format(df_out.columns[0]))
    plt.ylabel('{}'.format(df_out.columns[1]))
    plt.title('Dataset and outliers')
    
    
    #3D scatter plot of the dataset with outliers in red.
    if df.columns[2] == 'Labels_':
        pass
    else:
        plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter3D(df_out.iloc[:,0], df_out.iloc[:,1], df_out.iloc[:,2], color = 'k', alpha = 0.3)
        ax.scatter3D(df_out.iloc[:,0][outliers_], df_out.iloc[:,1][outliers_], df_out.iloc[:,2][outliers_], color = 'r')
        [ax.text(x, y, z, label) for x, y, z, label in zip(df_out.iloc[:,0], df.iloc[:,1], df_out.iloc[:,2], range(df_out.shape[0]))]
        ax.set_xlabel('{}'.format(df_out.columns[0]))
        ax.set_ylabel('{}'.format(df_out.columns[1]))
        ax.set_zlabel('{}'.format(df_out.columns[2]))
        ax.set_title('Dataset and outliers')
    
    return df_outliers, df

    

def ClusteringPerformance(df, clustering='hcpc', linkage_method='ward', n_clust=2, graph=True):
    
    if clustering == 'hcpc':
        condensed_dist_matx = pdist(df)
        Z = linkage(condensed_dist_matx, method=f'{linkage_method}')
        Inertia = sorted(Z[:,2], reverse=True)
        
        Model = AgglomerativeClustering(n_clusters = n_clust)
        Clusters = Model.fit_predict(df)

        
    elif clustering == 'kmean':
        K_MAX = int(10)
        KK = range(1,K_MAX+1)
        KM_ = [kmeans(df, k) for k in KK]
        centroids = [cent for (cent, var) in KM_]
        D_k = [cdist(df, cent, 'euclidean') for cent in centroids]
        dist = [np.min(dist_value, axis=1) for dist_value in D_k]
    
        tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
        totss = sum(pdist(df)**2)/df.shape[0]       # The total sum of squares
        betweenss = totss - tot_withinss          # The between-cluster sum of squares
        Inertia = betweenss/totss*100
        
        Model = KMeans(n_clusters = n_clust)
        Clusters = Model.fit_predict(df)

            
    if graph == True:
        fig, ax = plt.subplots(1,3, figsize=(12,4), tight_layout=True)
        
        
        #Figure1: Elbow plot
        ax[0].plot(np.arange(1, len(Inertia) + 1), Inertia, marker='o')
        ax[0].set_xlabel('Number of clusters')
        if clustering == 'kmean':
            ax[0].set_ylabel('Explained variance (%)')
        elif clustering == 'hcpc':
            ax[0].set_ylabel('Inertia')
        ax[0].set_title(f'Elbow plot for {clustering}')
        
        
        #Figure2: Silhouette plot
        silhouette_avg = silhouette_score(df, Clusters)
        sample_silhouette_values = silhouette_samples(df, Clusters)
        print(f'For n_clusters = {n_clust}, the average silhouette_score is : {silhouette_avg}')
        
        y_lower = 10
        for i in range(n_clust):
            ith_cluster_silhouette_values = sample_silhouette_values[Clusters == i]
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color_ = viridis_colors(n_clust)[i]
            ax[1].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                                facecolor=color_, edgecolor=color_, alpha=0.7)
            ax[1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            y_lower = y_upper + 10  # Compute the new y_lower for next plot (10 for the 0 samples)
        
        ax[1].set_ylim([0, len(df) + (n_clust + 1) * 10])
        ax[1].set_title("Silhouette plot")
        ax[1].set_xlabel("Silhouette coefficient")
        ax[1].set_ylabel("Cluster label")
        ax[1].axvline(x=silhouette_avg, color="red", linestyle="--")
        ax[1].set_yticks([])
        
        
        #Figure3: Mean silhouette score against clusters
        num_clusters = np.arange(2,11)
        SILHOUETTES = []
        for i in num_clusters:
            if clustering == 'hcpc':
                Ml = AgglomerativeClustering(n_clusters = i)
            elif clustering == 'kmean':
                Ml = KMeans(n_clusters = i)
            Cl = Ml.fit_predict(df)
            sil_mean = silhouette_score(df, Cl)
            SILHOUETTES.append(sil_mean)
        ax[2].plot(num_clusters, SILHOUETTES, marker='o')
        ax[2].set_title('Silhouette mean plot')
        ax[2].set_ylabel('Silhouette mean coeff')
        ax[2].set_xlabel('Number of clusters')
        
    return Model, Clusters, Inertia



def confidence_ellipse(x, y, ax, n_std=2.0, dataset_size=133, facecolor='none', **kwargs):
        
        '''
        
        Create a plot of the covariance confidence ellipse of *x* and *y*.
    
        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.
    
        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.
    
        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.
    
        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`
    
        Returns
        -------
        matplotlib.patches.Ellipse
        
        '''
        
        
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
    
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        cluster_size = x.size/dataset_size
        centroid_radius_x = np.sqrt(1 + pearson) * cluster_size
        centroid_radius_y = np.sqrt(1 - pearson) * cluster_size
    
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
        centroid = Ellipse((0, 0), width=centroid_radius_x * 2, height=centroid_radius_y * 2, facecolor=facecolor, **kwargs)
    
        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
    
        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
    
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
    
        ellipse.set_transform(transf + ax.transData)
        centroid.set_transform(transf + ax.transData)
        
        return ax.add_patch(ellipse), ax.add_patch(centroid)
    


def KMean(df_km, n_clust, pca_=False, explained_variance=None):
    
    '''
    
    Compute Kmeans on a dataset.
    
    df_km: DataFrame of shape (n_samples, n_variables)
        
    n_clust: int
        Number of arbitrary clusters.
    
    pca_: bool, default=True
        If True (resp. False), compute the Kmeans on a PCA-processed (resp. original) dataset.
    
    explained_variance: ndarray, default=None
        Use only if pca_=True. Explained variance for each principal component.
    
    Figures:
        -1: Elbow plot
        -2: 2D scatter plot with clusters
        -3: 3D scatter plot of the clusterized dataset
    
    Returns:    KM_clusters: ndarray
                    Array of int. Each int is the cluster's label each item belongs to.
                
                df_km: DataFrame of shape (n_samples, n_variables)
                    df_km with the clusters labels (KM_clusters) as last column.
    '''
       
    
    colors = viridis_colors(n_clust)

    #Drop nan rows and set Labels column
    df = df_km.dropna(axis = 0, how = 'all')
    if 'Labels_' in df_km.columns:
        df_km = df.drop(['Labels_'], axis = 1)
        labels = df['Labels_']


    Model, Clusters, Inertia = ClusteringPerformance(df_km, clustering='kmean', n_clust=n_clust, graph=False)
    
    
    #Figure1: 2D scatter plot with clusters
    fig, ax = plt.subplots(figsize=(5,4))
    
    [ax.scatter(df_km.iloc[i,0], df_km.iloc[i,1], color=colors[Clusters[i]], alpha=0.5) for i in range(df_km.shape[0])]
    
    # if 'Labels_' in df.columns:
    #     [ax.annotate(txt, (df_km.iloc[i,0], df_km.iloc[i,1])) for i, txt in enumerate(labels)]
    # else:
    #     [ax.annotate(txt, (df_km.iloc[i,0], df_km.iloc[i,1])) for i, txt in enumerate(range(df_km.shape[0]))]
        
        
    #Clusters ellipses   
    for i in range(n_clust):
        
        x = [df_km.iloc[j,0] for j in range(df_km.shape[0]) if Clusters[j] == i]
        y = [df_km.iloc[j,1] for j in range(df_km.shape[0]) if Clusters[j] == i]
        confidence_ellipse(np.array(x), np.array(y), ax=ax, dataset_size=df_km.shape[0], edgecolor=colors[i])
        
    if pca_ == True:
        ax.set_xlabel('{} ({:.2f}%)'.format(df_km.columns[0], explained_variance[0]*100))
        ax.set_ylabel('{} ({:.2f}%)'.format(df_km.columns[1], explained_variance[1]*100))
        ax.set_title('KMeans clustering on PCA')
        ax.axvline(x=0.,color='k',linestyle='--')
        ax.axhline(y=0.,color='k',linestyle='--')
    else:
        ax.set_xlabel('{}'.format(df_km.columns[0]))
        ax.set_ylabel('{}'.format(df_km.columns[1]))
        ax.set_title('KMeans clustering')
    
    
    #Figure3: 3D scatter plot of the clusterized dataset
    if df_km.shape[1] == 2:
        pass
    else:
       plt.figure()
       ax = plt.axes(projection='3d')
       
       [ax.scatter3D(df_km.iloc[i,0], df_km.iloc[i,1], df_km.iloc[i,2], color=colors[Clusters[i]], alpha=0.5) for i in range(df_km.shape[0])]
       [ax.text(x, y, z, label) for x, y, z, label in zip(df_km.iloc[:,0], df_km.iloc[:,1], df_km.iloc[:,2], range(df_km.shape[0]))]
       
       if pca_ == True:
           ax.set_xlabel('{} ({:.2f}%)'.format(df_km.columns[0], explained_variance[0]*100))
           ax.set_ylabel('{} ({:.2f}%)'.format(df_km.columns[1], explained_variance[1]*100))
           ax.set_zlabel('{} ({:.2f}%)'.format(df_km.columns[2], explained_variance[2]*100))
           ax.set_title('KMeans clustering on PCA')
       
       else:
           ax.set_xlabel('{}'.format(df_km.columns[0]))
           ax.set_ylabel('{}'.format(df_km.columns[1]))
           ax.set_zlabel('{}'.format(df_km.columns[2]))
           ax.set_title('KMeans clustering')
   
    
    df_km[f'{n_clust} Clusters'] = Clusters
    
    print('------------------')
    print('Kmeans - clusters: {}'.format(n_clust))
    print('------------------')
    # print(df_km)
    print('------------------')
    
    return Clusters, df_km



def HierAscClass(df_hc, n_clust, linkage_method='ward', pca_=False, explained_variance=None):
    
    '''
    
    Compute Ascending Hierarchical Classification (AHC) using the Ward method on a dataset.
    
    df_hc: DataFrame of shape (n_samples, n_variables)
        
    n_clust: int
        Set the threshold on the dendrogram to get the wanted number of clusters.
    
    pca_: bool, default=False
        If True (resp. False), compute the Kmeans on a PCA-processed (resp. original) dataset.
    
    explained_variance: ndarray, default=None
        Use only if pca_=True. Explained variance for each principal component.
    
    Figures:
        -1: Elbow plot of inertia
        -2: Dendrogram
        -3: 2D scatter plot with clusters
        -4: 3D scatter plot of the clusterized dataset
    
    Returns:    Clusters: ndarray
                    Array of int. Each int is the cluster's label each item belongs to.
                
                df_hc: DataFrame of shape (n_samples, n_variables)
                    df_km with the clusters labels (KM_clusters) as last column.
                    
    '''


    colors = viridis_colors(n_clust)

    #Drop nan rows and set Labels column
    df = df_hc.dropna(axis = 0, how = 'all')
    if 'Labels_' in df.columns:
        df_hc = df.drop(['Labels_'], axis = 1)
        labels = df['Labels_']
        
    condensed_dist_matx = pdist(df_hc)  #Condensed matrix of pairewise euclidean distances
    uncondensed_dist_matx = squareform(condensed_dist_matx)  #Uncondensed matrix of pairewise euclidean distances
    
    Z = linkage(condensed_dist_matx, method=f'{linkage_method}')
    c, coph_dists = cophenet(Z, condensed_dist_matx)
    print(f'Cophenetic Correlation Coefficient: {c}')
    
    Model, Clusters, Inertia = ClusteringPerformance(df_hc, clustering='hcpc', linkage_method=linkage_method, n_clust=n_clust, graph=False)
    
    #Figure1: Dendrogram.
    fig, ax = plt.subplots(figsize=(5,4))
    for i in range(len(Clusters)):
        set_link_color_palette(colors)
    dendrogram(Z, orientation='top', ax=ax, color_threshold=[(Inertia[i]-0.1) for i in range(len(colors)-1)][-1])
    ax.axhline(y = [(Inertia[i]-0.1) for i in range(len(colors)-1)][-1], color='r', linestyle='--')
    ax.set_ylabel('Inertia gain')

    if pca_ == True:
        ax.set_title('HCPC dendrogram on PCA')
    else:
        ax.set_title('HCPC dendrogram')
    

    #Figure2: 2D scatter plot with clusters.
    fig_scatter, ax_scatter = plt.subplots()
    [ax_scatter.scatter(df_hc.iloc[i,0], df_hc.iloc[i,1], color = colors[Clusters[i]], alpha = 0.5) for i in range(len(Clusters))]
    
    for i in range(n_clust):
        x = [df_hc.iloc[j,0] for j in range(df_hc.shape[0]) if Clusters[j] == i]
        y = [df_hc.iloc[j,1] for j in range(df_hc.shape[0]) if Clusters[j] == i]
        confidence_ellipse(np.array(x), np.array(y), ax=ax_scatter, dataset_size=df_hc.shape[0], edgecolor=colors[i])
    
    # if 'Labels_' in df.columns:
    #     [ax_scatter.annotate(txt, (df_hc.iloc[j,0], df_hc.iloc[j,1])) for j, txt in enumerate(labels)]
    # else:
    #     [ax_scatter.annotate(txt, (df_hc.iloc[j,0], df_hc.iloc[j,1])) for j, txt in enumerate(range(df_hc.shape[0]))]
        
    if pca_ == True:
        ax_scatter.set_xlabel('PC1 ({:.2f}%)'.format(explained_variance[0]*100))
        ax_scatter.set_ylabel('PC2 ({:.2f}%)'.format(explained_variance[1]*100))
        ax_scatter.set_title('HCPC on PCA')
        ax_scatter.axvline(x = 0.0, color = 'k', linestyle = '--')
        ax_scatter.axhline(y = 0.0, color = 'k', linestyle = '--')
    
    else:
        ax_scatter.set_xlabel('{}'.format(df_hc.columns[0]))
        ax_scatter.set_ylabel('{}'.format(df_hc.columns[1]))
        ax_scatter.set_title('HCPC')
        
    
    #Figure3: 3D scatter plot with clusters.
    if df_hc.shape[1] == 2:
        pass
    else:
        plt.figure()
        ax = plt.axes(projection='3d')
        
        [ax.scatter3D(df_hc.iloc[i,0], df_hc.iloc[i,1], df_hc.iloc[i,2], color=colors[Clusters[i]], alpha=0.5) for i in range(len(Clusters))]
        [ax.text(x, y, z, label) for x, y, z, label in zip(df_hc.iloc[:,0], df_hc.iloc[:,1], df_hc.iloc[:,2], range(df_hc.shape[0]))]
        
        if pca_ == True:
            ax.set_xlabel('PC1 ({:.2f}%)'.format(explained_variance[0]*100))
            ax.set_ylabel('PC2 ({:.2f}%)'.format(explained_variance[1]*100))
            ax.set_zlabel('PC3 ({:.2f}%)'.format(explained_variance[2]*100))
            ax.set_title('HCPC on PCA')
        
        else:
            ax.set_xlabel('{}'.format(df_hc.columns[0]))
            ax.set_ylabel('{}'.format(df_hc.columns[1]))
            ax.set_zlabel('{}'.format(df_hc.columns[2]))
            ax.set_title('HCPC')
            
    
    #Figure4: HCPC distance matrix
    fig_matrix = plt.figure(figsize=(6,6))
    ax1 = fig_matrix.add_axes([0.09,0.1,0.2,0.6])
    Z1 = dendrogram(Z, orientation='left', color_threshold=[(Inertia[i]-0.1) for i in range(len(colors)-1)][-1])
    ax2 = fig_matrix.add_axes([0.3,0.71,0.6,0.2])
    Z2 = dendrogram(Z, color_threshold=[(Inertia[i]-0.1) for i in range(len(colors)-1)][-1])
    axmatrix = fig_matrix.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    uncondensed_dist_matx = uncondensed_dist_matx[idx1,:]
    uncondensed_dist_matx = uncondensed_dist_matx[:,idx2]
    im = axmatrix.matshow(uncondensed_dist_matx, aspect='auto', origin='lower', cmap=cm.YlGnBu)
    axcolor = fig_matrix.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_title(f'Cophenetic coeff: {c}')
    ax2.set_xticks([])
    ax2.set_yticks([])
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    
    df_hc[f'{n_clust} Clusters'] = Clusters
    
    print('------------------')
    print('HCPC - clusters: {}'.format(n_clust))
    print('------------------')
    # print(df_hc)
    print('------------------')
    
    return Clusters, df_hc




def find_best_hyperparameters(model, model_name, param_grid, X_train, y_train, cross_val):
    
    '''
    
    Use GridSearchCV to find the best hyperparameters of the model.
    
    model: estimator object
        Estimator from Sklearn that calculates a score on a dataset.
    
    model_name: str
        Name of the estimator.
    
    param_grid: dict
        Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    
    X_train: DataFrame of shape (n_samples, n_variables)
    
    y_train: Series
        Series of labels
    
    cross_val: int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy. See the user guide on sklearn.
    
    Figure:
        Learning curve of train and validation sets.
    
    Returns:    modell: estimator
                    Estimator with best parameters
    
    '''
    
    
    grid = GridSearchCV(model, param_grid, cv = cross_val)
    grid.fit(X_train, y_train)
    
    print('------------------')
    print('Best model score')
    print(grid.best_score_)
    print('------------------')
    print('Best model parameters')
    print(grid.best_params_)
    
    modell = grid.best_estimator_
    
    N, train_score, val_score = learning_curve(modell, X_train, y_train, train_sizes = np.linspace(0.1, 1.0, 10), cv = cross_val)
    
    #Figure: Learning curve
    plt.figure()
    plt.plot(N, train_score.mean(axis=1), c = '#005172', label='train')
    plt.plot(N, val_score.mean(axis=1), c = '#0096d4', label='validation')
    plt.xlabel('Training data size')
    plt.ylabel('Score')
    plt.title('Learning curve for {}'.format(model_name))
    plt.legend()
    
    return modell
                   
                   


def analysis_window(dataframe, df_to_excel):
    
    '''
    
    dataframe: DataFrame of shape (n_samples, n_variables)
    
    '''
    
    sg.theme('DarkBlue')
    
    tab_pca = [[sg.Checkbox('PCA on', key='PCA on')],
               [sg.InputText(default_text='0.95', size=(5,1), key='threshold'),
                sg.Text('Threshold')],
               [sg.Button('RUN PCA')]]
    
    tab_tsne = [[sg.Checkbox('t-SNE on', key='tSNE_on')],
                [sg.InputText(default_text='5', size=(3,1), key='perplexity'),
                 sg.Text('Perplexity')],
                [sg.InputText(default_text='1000', size=(6,1), key='iterations'),
                 sg.Text('Iterations')],
                [sg.InputText(default_text='5', size=(6,1), key='num_fig'),
                 sg.Text('Amount of representations')],
                [sg.Button('RUN t-SNE')]]
    
    tab_clustering = [[sg.Checkbox('Kmeans', key='Kmeans')],
                      [sg.InputText(default_text='2', size=(3,1), key='kmean_clusters'),
                       sg.Text('Kmean clusters')],
                      [sg.Checkbox('Hierarchical clustering', key = 'HierAscClass')],
                      [sg.InputText(default_text='2', size=(3,1), key='Hcpc_clusters'),
                       sg.Text('HCPC clusters'),
                       sg.InputText(default_text='ward', size=(6,1), key='method'),
                       sg.Text('Linkage method')],
                      [sg.Button('PERFORMANCE EVALUATION')],
                      [sg.Button('RUN CLUSTERING')]]
    
    tab_learning = [[sg.Button('Select labels after clustering')],
                    [sg.Frame(layout=[
                    [sg.Checkbox('Train Test Split', key='TTS')],
                    [sg.Checkbox('Stratified Shuffle Split', key='SSS')],
                    [sg.InputText(default_text='0.2', size=(3,1), key='Test_size'),
                     sg.Text('Test size')],
                    [sg.InputText(default_text='20', size=(4,1), key='Split'),
                     sg.Text('Splits')]], title='Splitting method', relief=sg.RELIEF_SUNKEN)],
                    [sg.Frame(layout=[
                    [sg.Checkbox('Choose test dataset', key = 'arbitrary')],
                    [sg.InputText(size=(20,1), key='test_dataset'),
                     sg.FileBrowse()],
                    [sg.InputText(size=(8,1), key='labels_column'),
                     sg.Text('Labels column name')],
                    [sg.Checkbox('Pre-process test dataset', key='preprocess_test_dataset'),
                     sg.InputText(size=(8,1), default_text='median', key='imputer'),
                     sg.Text('Imputer method')]], title='Arbitrary test set', relief=sg.RELIEF_SUNKEN)],
                    [sg.Frame(layout=[
                    [sg.Checkbox('Random Forest', key='RandForest'),
                     sg.Checkbox('KNeighbors', key='KN')],
                    [sg.Checkbox('Support Vector Classifier', key='svc')]], title='Model', relief=sg.RELIEF_SUNKEN)],
                    [sg.Button('RUN LEARNING')]]
    
    analysis_layout = [[sg.Frame(layout=[
                       [sg.Checkbox('Supervised analysis', key='target')],
                       [sg.InputText(size=(7,1), key='target_text'),
                        sg.Text('Target column name')],
                       [sg.Frame('2D OR 3D DATASET', [
                       [sg.Text('Visualize dataset'),
                        sg.Button('GRAPH')]])],

                       [sg.Frame('DIMENSIONAL REDUCTION', [
                       [sg.TabGroup([[sg.Tab('PCA', tab_pca),
                                      sg.Tab('t-SNE', tab_tsne)]])]])],
                       [sg.Frame('ISOLATION FOREST', [
                       [sg.InputText(default_text='0.01', size=(4,1), key='contamination'),
                        sg.Text('Proportion of outliers')],
                       [sg.Checkbox('Outliers Off', key = 'outliers_off')],
                       [sg.Button('RUN ISOLATION FOREST')]])]], title='DIMENSIONAL PROCESSING', relief=sg.RELIEF_SUNKEN),
                        sg.Frame(layout=[
                            [sg.TabGroup([[sg.Tab('Clustering', tab_clustering),
                                           sg.Tab('Prediction', tab_learning)]])]], title='CLASSIFICATION', relief=sg.RELIEF_SUNKEN)],
                       [sg.Frame(layout=[
                            [sg.Text('Path'), sg.InputText(size=(30,1), key='save_dataset'), sg.FolderBrowse()],
                            [sg.Text('File name'), sg.InputText(size=(25,1), key='name')],
                            [sg.Button('SAVE')]], title='SAVING', relief=sg.RELIEF_SUNKEN)]]
              
    
    analysis_window = sg.Window('CLUSTERING ANALYSIS', analysis_layout, location=(0,0))
    
    
    while True:
        
       analysis_event, analysis_value = analysis_window.read()
       
       try:
           if analysis_event in (None, 'Close'):
               break
           
            
           if analysis_value['target'] == True:
               y = dataframe[analysis_value['target_text']]
               dataframe = dataframe.drop([analysis_value['target_text']], axis=1)
               
               
               if y.dtype == 'object':
                   
                   encoder = LabelEncoder()
                   y = encoder.fit_transform(y)
               
                          
           if analysis_event == 'GRAPH':
               plt.figure()
               plt.scatter(dataframe.iloc[:,0], dataframe.iloc[:,1], color = 'k', alpha = 0.4)
               [plt.annotate(i, (dataframe.iloc[i,0], dataframe.iloc[i,1])) for i, txt in enumerate(range(dataframe.shape[0]))]
               plt.xlabel('{}'.format(dataframe.columns[0]))
               plt.ylabel('{}'.format(dataframe.columns[1]))
               plt.title('Dataset')
               
               if dataframe.shape[1] == 3:
                   plt.figure()
                   ax = plt.axes(projection='3d')
                   ax.scatter3D(dataframe.iloc[:,0], dataframe.iloc[:,1], dataframe.iloc[:,2], color = 'k', alpha = 0.4)
                   [ax.text(x, y, z, label) for x, y, z, label in zip(dataframe.iloc[:,0], dataframe.iloc[:,1], dataframe.iloc[:,2], range(dataframe.shape[0]))]
                   ax.set_xlabel('{}'.format(dataframe.columns[0]))
                   ax.set_ylabel('{}'.format(dataframe.columns[1]))
                   ax.set_zlabel('{}'.format(dataframe.columns[2]))
                   ax.set_title('Dataset')
                   
                   
           if analysis_event == 'RUN PCA':
               ExpVar, df_eigen = PrinCompAn(dataframe, float(analysis_value['threshold']))
               
            
           if analysis_event == 'RUN t-SNE':
               if analysis_value['target'] == True:
                   Tsne(dataframe, int(analysis_value['perplexity']), int(analysis_value['iterations']), np.array(y), int(analysis_value['num_fig']))
               
               else:
                   Tsne(dataframe, int(analysis_value['perplexity']), int(analysis_value['iterations']), num_fig = int(analysis_value['num_fig']))
                     
           
           if analysis_event == 'RUN ISOLATION FOREST':
                if analysis_value['contamination'] == str('auto'):
                    n_contamination = 'auto'
                
                else:
                    n_contamination = float(analysis_value['contamination'])
                    
                    
                if analysis_value['PCA on'] == True:
                    if analysis_value['outliers_off'] == True:
                        df_outliers, df_eigen = outliers_detection(df_eigen, n_contamination, analysis_value['outliers_off'])

                    else:
                        outliers_detection(df_eigen, n_contamination, analysis_value['outliers_off'])
                
                else:
                    if analysis_value['outliers_off'] == True:
                        df_outliers, df_iso = outliers_detection(dataframe, n_contamination, analysis_value['outliers_off'])
                        
                    else:
                        outliers_detection(dataframe, n_contamination, analysis_value['outliers_off'])
                    
                   
           if analysis_event == 'PERFORMANCE EVALUATION':
               if analysis_value['Kmeans'] == True:
                   if analysis_value['PCA on'] == True:
                       ClusteringPerformance(df_eigen, clustering='kmean', n_clust=int(analysis_value['kmean_clusters']))
                   else:
                       if analysis_value['outliers_off'] == True:
                           ClusteringPerformance(df_iso, clustering='kmean', n_clust=int(analysis_value['kmean_clusters']))
                       else:
                           ClusteringPerformance(dataframe, clustering='kmean', n_clust=int(analysis_value['kmean_clusters']))
                           
               if analysis_value['HierAscClass'] == True:
                   if analysis_value['PCA on'] == True:
                       ClusteringPerformance(df_eigen, clustering='hcpc', linkage_method=analysis_value['method'], n_clust=int(analysis_value['Hcpc_clusters']))
                   else:
                       if analysis_value['outliers_off'] == True:
                           ClusteringPerformance(df_iso, clustering='hcpc', linkage_method=analysis_value['method'], n_clust=int(analysis_value['Hcpc_clusters']))
                       else:
                           ClusteringPerformance(dataframe, clustering='hcpc', linkage_method=analysis_value['method'], n_clust=int(analysis_value['Hcpc_clusters']))
                       
                   
                   
           if analysis_event == 'RUN CLUSTERING':
               if analysis_value['Kmeans'] == True:
                   if analysis_value['PCA on'] == True:
                       KM_clusters, df_clust = KMean(df_eigen, int(analysis_value['kmean_clusters']), analysis_value['PCA on'], ExpVar)
                       
                   else:
                       if analysis_value['outliers_off'] == True:
                           KM_clusters, df_clust = KMean(df_iso, int(analysis_value['kmean_clusters']), analysis_value['PCA on'])
                           
                       
                       else:
                           KM_clusters, df_clust = KMean(dataframe, int(analysis_value['kmean_clusters']), analysis_value['PCA on'])
                           dataframe = dataframe.drop(f'{analysis_value[kmean_clusters]} Clusters', axis=1)
                    
                    
                   if analysis_value['tSNE_on'] == True:
                       Tsne(dataframe, int(analysis_value['perplexity']), int(analysis_value['iterations']), KM_clusters)
                           
                   
                    
               if analysis_value['HierAscClass'] == True:
                   if analysis_value['PCA on'] == True:
                       HCPC_clusters, df_clust = HierAscClass(df_eigen, int(analysis_value['Hcpc_clusters']), analysis_value['method'], analysis_value['PCA on'], ExpVar)
                        
                   else:
                       if analysis_value['outliers_off'] == True:
                           HCPC_clusters, df_clust = HierAscClass(df_iso, int(analysis_value['Hcpc_clusters']), analysis_value['method'], analysis_value['PCA on'])
                       
                       else:
                           HCPC_clusters, df_clust = HierAscClass(dataframe, int(analysis_value['Hcpc_clusters']), analysis_value['method'], analysis_value['PCA on'])
                           dataframe = dataframe.drop(f'{analysis_value[Hcpc_clusters]} Clusters', axis=1)
                       
                        
                   if analysis_value['tSNE_on'] == True:
                       Tsne(dataframe, int(analysis_value['perplexity']), int(analysis_value['iterations']), HCPC_clusters)
               
               
               dataframe = pd.concat([dataframe, df_clust.iloc[:,-1]], axis=1)
               
               if analysis_value['outliers_off'] == True:
                   outliers = dataframe.iloc[df_outliers.index, :]
                   dataframe = dataframe.drop(df_outliers.index, axis = 0)

                   print('------------------')
                   print('Outliers from original dataset')
                   print('------------------')
                   print(outliers)
                   print('------------------')
               
               print('------------------')
               print('Dataset clusterized')
               print('------------------')
               print(dataframe)
               print('------------------')
               
               
               dataframe_for_excel_file = pd.concat([df_to_excel, df_clust.iloc[:,-1]], axis=1)
               
               
           if analysis_event == 'Select labels after clustering':
               dataframe = LabelsSelection(dataframe)

           if analysis_event == 'RUN LEARNING':
               if analysis_value['target'] == False:
                   y = dataframe['Cluster']
                   dataframe = dataframe.drop(['Cluster'], axis=1)
                   
                   
               if analysis_value['RandForest'] == True:
                   model = RandomForestClassifier(random_state = 5)
                   model_name = 'RandForestClass'
                   
                   param_grid = {'n_estimators': np.arange(1, 20), 'max_depth': np.arange(1,20)}
                   
               
               if analysis_value['KN'] == True:
                   model = KNeighborsClassifier(metric='euclidean')
                   model_name = 'KNeighbors'
                   
                   param_grid = {'n_neighbors': np.arange(1, 20)}

            
               if analysis_value['svc'] == True:
                   model = SVC()
                   model_name = 'SVC'
                   
                   param_grid = {'C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]}
               
                   
               if analysis_value['TTS'] == True:
                   X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=float(analysis_value['Test_size']), random_state=5)
                   
                   modell = find_best_hyperparameters(model, model_name, param_grid, X_train, y_train, int(analysis_value['Split']))
                   
                   print('------------------')
                   print('Score on test set')
                   print(modell.score(X_test, y_test))
                   print('------------------') 
                   
                   matrix = confusion_matrix(y_test, modell.predict(X_test), normalize = 'true')
                   
                   print('Confusion matrix')
                   print(matrix)
                   print('------------------')
                   
                   fig, ax = plt.subplots()
                   sns.heatmap(matrix, cmap='YlGnBu', vmin=0., vmax=1.,
                               xticklabels=[f'Class {i}' for i in range(len(np.unique(y)))],
                               yticklabels=[f'Class {i}' for i in range(len(np.unique(y)))],
                               ax=ax, annot=True, square=True)
                   ax.set_xlabel('Predicted')
                   ax.set_ylabel('Actual')
                   ax.set_title('Confusion matrix')
                   
                   
                   
               if analysis_value['SSS'] == True:
                   splitting = StratifiedShuffleSplit(n_splits = int(analysis_value['Split']), test_size = float(analysis_value['Test_size']), random_state = 5)
                   
                   
                   confusion_mats_all = []
                   score_all = []
                    
                   for (train, test), i in zip(splitting.split(dataframe, y), range(1, int(analysis_value['Split'])+1)):
                        
                       X_train, X_test = dataframe.iloc[train], dataframe.iloc[test]
                       y_train, y_test = y.iloc[train], y.iloc[test]
                       
                       modell = find_best_hyperparameters(model, model_name, param_grid, X_train, y_train, int(analysis_value['Split']))
                       
                       print('Test score')
                       print(modell.score(X_test, y_test))
                       print('------------------')
                       
                       matrix = confusion_matrix(y_test, modell.predict(X_test), normalize = 'true')
                       
                       confusion_mats_all.append(pd.DataFrame(matrix))
                       score_all.append(modell.score(X_test, y_test))
        
                   confusion_mat_final = pd.concat(confusion_mats_all).groupby(level=0).mean()
                   
                   fig, ax = plt.subplots()
                   sns.heatmap(confusion_mat_final, cmap='YlGnBu', vmin=0., vmax=1.,
                               xticklabels=[f'Class {i}' for i in range(len(np.unique(y)))],
                               yticklabels=[f'Class {i}' for i in range(len(np.unique(y)))],
                               ax=ax, annot=True, square=True)
                   ax.set_xlabel('Predicted')
                   ax.set_ylabel('Actual')
                   ax.set_title('Confusion matrix')
               
               
                
               if analysis_value['arbitrary'] == True:
                   test_set = pd.read_excel('{}'.format(analysis_value['test_dataset']), header=0)
                   
                   y_test = test_set[analysis_value['labels_column']]
                   X_test = test_set.drop(['{}'.format(test_set.columns[i]) for i in range(len(test_set.columns)) if test_set.columns[i] not in dataframe.columns], axis=1)
                   
                   if analysis_value['preprocess_test_dataset'] == True:

                       encoder = LabelEncoder()
                       scaler = StandardScaler()
                        
                       for col in X_test.columns:
                            
                           if X_test[col].dtype != 'object':
                               imputer = SimpleImputer(strategy = analysis_value['imputer']).fit_transform(X_test[col].values.reshape(-1,1))
                               
                               scaling = scaler.fit_transform(imputer).reshape(-1,)
                               X_test = X_test.replace(np.array(X_test[col]), scaling)
                           
                           if X_test[col].dtype == 'object':
                               
                               rep = np.array(encoder.fit_transform(X_test[col]))
                               X_test = X_test.replace(np.array(X_test[col]), rep)
                       
                   print('------------------')
                   print('Test dataset')
                   print(X_test)
                   print(y_test)
                   print('------------------')
                   
                   modell = find_best_hyperparameters(model, model_name, param_grid, dataframe, y, int(analysis_value['Split']))

                   print('------------------')
                   print('Test score')
                   print(modell.score(X_test, y_test))
                   print('------------------')
                       
                   matrix = confusion_matrix(y_test, modell.predict(X_test), normalize = 'true')
                  
                   print('------------------')
                   print('Confusion matrix')
                   print(matrix)
                   print('------------------')
                   
                   fig, ax = plt.subplots()
                   sns.heatmap(matrix, cmap='YlGnBu', vmin=0., vmax=1.,
                               xticklabels=[f'Class {i}' for i in range(len(np.unique(y)))],
                               yticklabels=[f'Class {i}' for i in range(len(np.unique(y)))],
                               ax=ax, annot=True, square=True)
                   ax.set_xlabel('Predicted')
                   ax.set_ylabel('Actual')
                   ax.set_title('Confusion matrix')
                   
                       
           if analysis_event == 'SAVE':
               if analysis_value['outliers_off'] == True:
                   with pd.ExcelWriter('{}\{}_outliers.xlsx'.format(analysis_value['save_dataset'], analysis_value['name'])) as writer:
                       outliers.to_excel(writer)
                   
               
               with pd.ExcelWriter('{}\{}.xlsx'.format(analysis_value['save_dataset'], analysis_value['name'])) as writer:
                   dataframe_for_excel_file.to_excel(writer)
                   
               print('DONE')
               
              
       except:
           pass
       
    analysis_window.close()




def variables(tab):
    
    '''
    
    tab: dict
        Dictionary with excel sheet names (str) as keys and DataFrame of variables as values.
    
    Returns:    df_variables: DataFrame of shape (n_samples, n_variables)
    
    '''
    
    tabs_checkbox = {}
    
    for i in tab:
        tabs_checkbox[i] = []
    
        for j in range(len(tab[i].columns)):
            tabs_checkbox[i].append([sg.Checkbox(f'{tab[i].columns[j]}', key=f'{i}_{tab[i].columns[j]}')])
            
    sg.theme('DarkBlue')
    
    tab_layout = [[sg.Frame(layout=[
                  [sg.TabGroup([[sg.Tab(sheet, checkbox_list) for sheet, checkbox_list in tabs_checkbox.items()]], key='current_sheet')],
                  [sg.Button('Select all', key='select_all'),sg.Button('Deselect all', key='deselect_all')]], title='VARIABLES SELECTION', relief=sg.RELIEF_SUNKEN)],
                  [sg.Frame(layout=[
                  [sg.InputText(default_text = 'median', size=(7,1), key='Imputer'), sg.Text('Imputer strategy')],
                  [sg.Checkbox('Pre-processing', key='preprocessing', default=True)]], title='PREPROCESSING', relief=sg.RELIEF_SUNKEN)],
                  [sg.Button('ANALYSE')]]
    
    window = sg.Window('VARIABLES').Layout([[sg.Column(tab_layout, size=(300,900), scrollable=True)]])
    
    while True:
        
        event, value = window.read()
        
        try:
            if event in (None, 'Close'):
                break
            
            if event == 'select_all':
                current_heet = value['current_sheet']
                checkbox_list_of_list = tabs_checkbox[current_heet]
                for checkbox_list in checkbox_list_of_list:
                    for checkbox in checkbox_list:
                        window[checkbox.key].update(True)
                    
            if event == 'deselect_all':
                current_heet = value['current_sheet']
                checkbox_list_of_list = tabs_checkbox[current_heet]
                for checkbox_list in checkbox_list_of_list:
                    for checkbox in checkbox_list:
                        window[checkbox.key].update(False)
            
            if event == 'ANALYSE':
                
                df_variables = pd.DataFrame(index=None, columns=None)
                df_to_excel = pd.DataFrame(index=None, columns=None)
                
                for sheet, checkbox_list in tabs_checkbox.items():
                    
                    for item in range(len(checkbox_list)):
                        
                        if value[f'{sheet}_{tab[sheet].columns[item]}'] == True:
                            
                            df_to_excel[f'{tab[sheet].columns[item]}'] = tab[sheet][tab[sheet].columns[item]]

                            if tab[sheet][tab[sheet].columns[item]].dtype == 'object':
                                
                                if value['preprocessing'] == True:
                                    
                                    encoder = LabelEncoder()
                                    targets = encoder.fit_transform(tab[sheet][tab[sheet].columns[item]])
                                
                                else:
                                    
                                    targets = tab[sheet][tab[sheet].columns[item]]
                                    
                                df_variables[f'{tab[sheet].columns[item]}'] = targets
                                
                            
                            elif  tab[sheet].columns[item] == 'Cluster':
                                
                                df_variables[f'{tab[sheet].columns[item]}'] = tab[sheet][tab[sheet].columns[item]]
                                
                            
                            else:
                                
                                if value['preprocessing'] == True:
                                    
                                    var = np.array(tab[sheet][tab[sheet].columns[item]]).reshape(-1,1)
                                    
                                    imputer = SimpleImputer(strategy=value['Imputer']).fit_transform(var)
                                    
                                    scaler = StandardScaler()
                                    var_ = scaler.fit_transform(imputer).reshape(-1,)
                                
                                else:
                                    
                                    var_ = tab[sheet][tab[sheet].columns[item]]
                                    
                                df_variables[f'{tab[sheet].columns[item]}'] = var_
                
            
                # print('Dataframe used for analyses')
                # print('------------------')
                # print(df_to_excel)
                 
                analysis_window(df_variables, df_to_excel)
             
        except:
            pass
    window.close()
    
    return(df_variables)




def LabelsSelection(dataframe):
    
    clusters = []
    
    for j in range(len(dataframe.columns)):
        if 'Clusters' in dataframe.columns[j]:
            clusters.append(f'{dataframe.columns[j]}')
            
            
    sg.theme('DarkBlue')
    
    tab_layout = [[sg.Checkbox(f'{dataframe.columns[col]}') for col in range(len(dataframe.columns)) if 'Clusters' in dataframe.columns[col]],
                  [sg.Button('DONE')]]
    
    window = sg.Window('LABELS SELECTION').Layout([[sg.Column(tab_layout)]])
    
    while True:
        
        event_labels, value_labels = window.read()
        
        try:
            if event_labels in (None, 'Close'):
                break
            
            if event_labels == 'DONE':
                for item in range(len(clusters)):
                    if value_labels[item] == True:
                        Cluster = dataframe[clusters[item]]
                
                dataframe = dataframe.drop([i for i in dataframe.columns if 'Clusters' in i], axis=1)
                dataframe['Cluster'] = Cluster
                # print(dataframe)        

        except:
            pass
    window.close()
    
    return dataframe
    
    




if __name__ == '__main__':
    
    import PySimpleGUI as sg
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    from mpl_toolkits import mplot3d
    from mlxtend.plotting import plot_pca_correlation_graph
    from scipy.cluster.hierarchy import cophenet, dendrogram, linkage, set_link_color_palette
    from scipy.cluster.vq import kmeans
    from scipy.spatial.distance import cdist, pdist, squareform
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.impute import SimpleImputer
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split, learning_curve
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import confusion_matrix

    sg.theme('DarkBlue')
        
    main_layout = [[sg.Text('File path')],
                   [sg.InputText(size=(35,1)), sg.FileBrowse()],
                   [sg.Button('GO')]]
              
    main_window = sg.Window('ClusterGUI', main_layout, location=(0,0))
    
    while True:
       main_event, main_value = main_window.read()
       try:
           if main_event in (None, 'Close'):
               plt.close('all')
               break
           
           if main_event == 'GO':
               Dataset = {}
               
               file_sheets = pd.ExcelFile(str(main_value[0])).sheet_names
               for i in file_sheets:
                   data = pd.read_excel(f'{main_value[0]}', sheet_name=i, header=0)
                   Dataset[f'{i}'] = data

               variables(Dataset)
               
       except:
           pass
    main_window.close()