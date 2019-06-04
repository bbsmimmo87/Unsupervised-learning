# Unsupervised-learning- Kmeans
from sklearn.cluster import KMeans

model=KMeans(n_clusters=3)
model.fit(samples)

    [out] - KMeans(algorithm='auto',...)
    
labels=model.predict(samples)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]

centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)

plt.show()

# Scatter plots
import matplotlib.pyplot as plt

xs=samples[:,0]

ys=samples[:,2]

plt.scatter(xs, ys, c=labels,alpha=0.5)

plt.show()

# Evaluating a clustering

***Cross tabulation with pandas =pd.crosstab***

***print(model.inertia_), we can see how far they spread out. lower is better***

***df = pd.DataFrame({'labels':labels ,'species': species})***

ks = range(1, 6)
inertias = []

for k in ks:
    #Create a KMeans instance with k clusters: model
    model=KMeans(n_clusters=k)
    
    #Fit model to samples
    model.fit(samples)
    
    #Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
#Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# An Example

#Create a KMeans model with 3 clusters: model
pdmodel = KMeans(n_clusters=3)

#Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

#Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

#Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

#Display ct
print(ct)

#   StandardScaler

from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
scaler.fit(samples)
samples_scaled = scaler.transform(samples)

- Use fit()/transform() with StandardScaler
- Use fit()/predict() with KMeans

# Pipelines combine multiple steps

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
Kmeans = KMeans(n_clusters=3)
from sklearn.pipline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
labels = pipeline.predict(samples)

# Normalizer

from sklearn.preprocessing import Normalizer

normalizer = Normalizer()

kmeans = KMeans(n_clusters=10)

pipeline = make_pipeline(normalizer,kmeans)

pipeline.fit(movements)
