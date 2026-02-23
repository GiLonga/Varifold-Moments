### This code is to construct and transform polygons as 
### lists of complex numbers. You can draw the polygons with
### the function Draw defined at the end of the file.
### Try, for instance, Draw(regular_gon(6)) or Draw(Spike(regular_gon(4))).

### Constructing and Transforming Polygons ###
## Warning: functional programming style

import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from umap import UMAP
from sklearn.decomposition import PCA
from  scipy.interpolate import interp1d

## Some Basic Polygons

def expi(angle):
    return complex(math.cos(angle), math.sin(angle))

def regular_gon(n, angle=0):
    "Creates a regular n-gon"
    res = []
    for k in range(n):
        res.append(expi(k*2*math.pi/n + angle))
    return tuple(res)    

def RandomTriangle(a=-4,b=4):
    """ Creates a random triangle of base 4 and height 2 with 
    apex 2+uj, u uniformly choisen from the interval [a,b]."""
    u = random.uniform(a,b)
    return (-2,2,complex(u,2))

def RandomParallelogram(a=-4,b=4):
    u = random.uniform(a,b)
    v = random.uniform(a,b)
    return (0, 2, complex(u,v), complex(u-2,v))
    

## Basic Geometric Transformations

def Rotate(angle,polygon):
    "Rotates polygon counter-clockwise by angle given in radians"
    return tuple(map(lambda z: expi(angle)*z,polygon))

def linear(a,b,c,d):
    "Returns the linear transformation given by the matrix ((a,b),(c,d))"
    return lambda z: complex(a*z.real + b*z.imag, c*z.real + d*z.imag)

def Linear(a,b,c,d,polygon):
    "Applies the linear transformation ((a,b),(c,d)) to the polygon"
    return tuple(map(linear(a,b,c,d),polygon))
    

## Other Transformations


def Subdivide(atuple, func= lambda x,y: (x+y)/2): 
    res = []
    n = len(atuple)
    for k in range(n-1):
        res.append(atuple[k])
        res.append(func(atuple[k],atuple[k+1]))
    res.append(atuple[n-1])    
    res.append(func(atuple[n-1],atuple[0]))  
    return tuple(res)

def ReSubdivide(polygon,n=3):
    X = polygon 
    for k in range(n):
        X = Subdivide(X)
    return X 


def Spike(polygon, parameter=-0.2):
    return Subdivide(polygon,lambda x,y: parameter*1j*(y-x) + (x+y)/2)

## Random Polygons and Random Perturbations

def RandomPoint(lower_left, upper_right):
    """Constructs a random point (a complex number) inside a  a rectangle 
    given by its lower left and upper right corners."""
    x = random.uniform(lower_left.real, upper_right.real)
    y = random.uniform(lower_left.imag, upper_right.imag)
    return complex(x,y)

def RandomPolygon(lower_left, upper_right,n):
    """Constructs an n-sided polygon inside a rectangle given by its lower 
    left and upper right corners."""
    acc = [ ]
    for k in range(n):
        point = RandomPoint(lower_left, upper_right)
        acc.append(point)
    return tuple(acc)   

def Perturb(X,epsilon=0.05):
    "Returns a slightly perturbed version of polygon X and normalizes it"
    acc = []
    for x in X:
        point = complex(random.uniform(-epsilon,epsilon),
                        random.uniform(-epsilon,epsilon))
        acc.append(x+point)
    return Normalize(tuple(acc)) 

def RePerturb(polygon,n=2):
    X = polygon 
    for k in range(n):
        X = Perturb(X)
    return X 

### Convex hull of a polygon

### Minkowski sum of two convex polygons

### Normalizing polygons

## Computing the area using Stokes theorem.
def CyclicSum(func, atuple):
    """Given a function of two variables func(x,y) and a list with at least three
    entries, alist  = (a1 a2 a3 ... an), this function computes the cyclic sum 
    f(a1,a2) + f(a2,a3)  + ... + f(an,a1)."""
    acc = func(atuple[-1], atuple[0])
    for x in range(len(atuple) - 1):
        acc = acc + func(atuple[x],atuple[x+1])
    return acc 

def pre_area(z,w):
    return (w-z) * (w.conjugate() + z.conjugate())/2

def area(X):
    a = CyclicSum(pre_area,X)
    return a.imag / 2

## Computing the center of mass

def pre_cm(z,w):
    return ((w-z)/3) * (abs(z)**2 + (z*w.conjugate() + w*z.conjugate())/2 
                       + abs(w)**2)

def CenterMass(X):
    "Center of mass of polygon X"
    return CyclicSum(pre_cm,X)/ (area(X)*(0+2j))

def Normalize(X):
    """Displaces a polygon so that its center of mass lies at the origin
    and dilates it so its area equals 10"""
    CM = CenterMass(X)
    a = math.sqrt(area(X)/10)
    return tuple(map(lambda x: (x-CM)/a, X))


def features(polygon):
    "Feature space for polygons consisting of four basic invariants"
    length = Moment(0,0,0,polygon).real
    length_centermass = abs(Moment(1,0,0,polygon))
    length_rotational_moment = Moment(1,1,0,polygon).real
    area_rotational_moment = Moment(1,2,1,polygon).imag
    return (length, length_centermass, length_rotational_moment,
                 area_rotational_moment)

def features2(polygon):
    length = Moment(0,0,0,polygon).real
    length_centermass = abs(Moment(1,0,0,polygon))
    axial_symmetry = Moment(0,2,2,polygon).imag
    affine_invariant = affine(polygon)
    return (length,length_centermass,axial_symmetry,affine_invariant)

def clean(string,k):
    "This is used to format the features for the csv file"
    acc = ''
    for x in string:
        if x != ' ' and x != '(' and x != ')':
           acc = acc + x
    acc = acc + ',' + str(k)       
    return acc





## Plotting a closed polygon that is given as a tuple of complex numbers.

def Real(polygon):
    "List of x cooordinates of points in polygon"
    X = [z.real for z in polygon]
    X.append(X[0])
    return X

def Imag(polygon):
    "List of y coordinates of points in polygon"
    X = [z.imag for z in polygon]
    X.append(X[0])
    return X

def Draw(polygon):
    plt.grid("on")
    plt.rcParams['figure.figsize'] = [10, 10] # for square canvas
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.plot(Real(polygon), Imag(polygon), linewidth=4, color='black')

    ### From Scheme to Python : moments.ss 

import numpy as np

def CyclicSum(func, atuple):
    """Given a function of two variables func(x,y) and a list with at least three
    entries, alist  = (a1 a2 a3 ... an), this function computes the cyclic sum 
    f(a1,a2) + f(a2,a3)  + ... + f(an,a1)."""
    acc = func(atuple[-1], atuple[0])
    for x in range(len(atuple) - 1):
        acc = acc + func(atuple[x],atuple[x+1])
    return acc 


def sort(atuple):
    """ sorting a tuple without side effects."""
    return tuple(sorted(atuple))

from functools import reduce

def factorial(n):
    return reduce(lambda x,y:x*y, range(1,n+1),1)

def choose(n,k):
    return int(reduce(lambda x,y:x*y, range(n-k+1 ,n+1),1)/factorial(k))

def sign(x):
    if x % 2 == 0:
        return 1
    else:
       return -1
   
## Reminder: in Python complex numbers have the form a + bj. Their real part
## is given as attributes z.real, z.imag, and conjugation is a method: 
## z.conjugate(). The module is given by abs(z), and their powers are computed
## as usual: z**n (you can also write pow(z,n)).

## Computing the pqr moments of a segment given as zw 
## with z and w complex numbers. This is a triple sum that it is best 
## to dissect into an inner sum and an double sum after that.

## First compute the innermost sum in the triple sum.
def innermost(p,q,z,w,k,l):
    zconj = z.conjugate()
    wconj = w.conjugate()
    acc = 0
    for s in range(k+l+1):
       acc = acc + (pow(z,p-k) * pow(zconj, q-l) * pow(w,k) * pow(wconj,l) 
       * sign(s) *  choose(k+l,s)) / (p+q-k-l+s+1)
    return acc 

## Now we can easily write the triple sum.

def MomentTripleSum(p,q,r,z,w):
    """The pqr moment of the segment zw, where z and w are complex numbers"""
    multiplier = pow(w-z,r)/pow(abs(w-z),r-1)
    acc = 0
    for k in range(0,p+1):
        for l in range(0,q+1):
            acc = acc + choose(p,k) * choose(q,l) * innermost(p,q,z,w,k,l)
    return multiplier * acc
     

def Moment(p,q,r,X):
    """The pqr moment of closed polygon X given as a tuple of complex numbers"""
    return CyclicSum(lambda z,w: MomentTripleSum(p,q,r,z,w), X)

def affine(X):
    "The simplest affine invariant coming from moments"
    x = Moment(2,1,1,X).real
    y = Moment(2,1,1,X).imag
    z = Moment(1,2,1,X).imag
    return z*z - 4*(x*x + y*y)



### Normalizing polygons

## Computing the area using Stokes theorem.

def pre_area(z,w):
    return (w-z) * (w.conjugate() + z.conjugate())/2

def area(X):
    a = CyclicSum(pre_area,X)
    return a.imag / 2

## Computing the center of mass

def pre_cm(z,w):
    return ((w-z)/3) * (abs(z)**2 + (z*w.conjugate() + w*z.conjugate())/2 
                        + abs(w)**2)

def CenterMass(X):
    "Center of mass of polygon X"
    return CyclicSum(pre_cm,X)/ (area(X)*(0+2j))

def Centering(X):
    "Displacing a polygon so that its center of mass lies at the origin"
    CM = CenterMass(X)
    return tuple(map(lambda x: x-CM, X))

## 23/05/2025  There is a problem with Centering and CenterMass. You must
## be very careful the polygons are oriented possitively.


## Plotting a closed polygon that is given as a tuple of complex numbers.


def Real(polygon):
    "List of x cooordinates of points in polygon"
    X = [z.real for z in polygon]
    X.append(X[0])
    return X
    
def Imag(polygon):
    "List of y coordinates of points in polygon"
    X = [z.imag for z in polygon]
    X.append(X[0])
    return X

def Draw(polygon):
    plt.grid("on")
    plt.rcParams['figure.figsize'] = [10, 10] # for square canvas
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.plot(Real(polygon), Imag(polygon), linewidth=4, color='black')
    
## It would be nice to make the polygonal line slightly thicker.   
    
   ### Constructing and Transforming Polygons ###





## Some Basic Polygons

def expi(angle):
       return complex(math.cos(angle), math.sin(angle))

def regular_gon(n, angle=0):
       "Creates a regular n-gon"
       res = []
       for k in range(n):
           res.append(expi(k*2*math.pi/n + angle))
       return tuple(res)    

def RandomTriangle(a=-4,b=4):
       """ Creates a randome triangle of base 4 and height 2 with 
       apex 2+uj, u uniformly choisen from the interval [a,b]."""
       u = random.uniform(a,b)
       return (-2,2,complex(u,2))

def RandomParallelogram(a=-4,b=4):
       u = random.uniform(a,b)
       v = random.uniform(a,b)
       return (0, 2, complex(u,v), complex(u-2,v))
       

   ## Basic Geometric Transformations

def Rotate(angle,polygon):
       "Rotates polygon counter-clockwise by angle given in radians"
       return tuple(map(lambda z: expi(angle)*z,polygon))

def linear(a,b,c,d):
       "Returns the linear transformation given by the matrix ((a,b),(c,d))"
       return lambda z: complex(a*z.real + b*z.imag, c*z.real + d*z.imag)

def Linear(a,b,c,d,polygon):
       "Applies the linear transformation ((a,b),(c,d)) to the polygon"
       return tuple(map(linear(a,b,c,d),polygon))
   
def RandomLinear(polygon,epsilon=0.1):
    b = random.uniform(-epsilon,epsilon)
    c = random.uniform(-epsilon,epsilon)
    return Normalize(tuple(map(linear(1,b,c,1),polygon)))

def Projective(a,b,c,d,e,f):
    return lambda z: complex((z.real + a*z.imag + b)/(e*z.real + 
                                                      f*z.imag + 1),
                             (c*z.real + z.imag + d)/(e*z.real + f*z.imag + 1))
    

# Should be imporoved to take into account that points may be sent to infinity    
def RandomProjective(polygon,epsilon=0.05):
    a = random.uniform(-epsilon,epsilon)
    b = random.uniform(-epsilon,epsilon)
    c = random.uniform(-epsilon,epsilon)
    d = random.uniform(-epsilon,epsilon)
    e = random.uniform(-epsilon,epsilon)
    f = random.uniform(-epsilon,epsilon)
    return Normalize(tuple(map(Projective(a,b,c,d,e,f),polygon)))
      

## Other Transformations


def Subdivide(atuple, func= lambda x,y: (x+y)/2): 
    res = []
    n = len(atuple)
    for k in range(n-1):
        res.append(atuple[k])
        res.append(func(atuple[k],atuple[k+1]))
        res.append(atuple[n-1])    
        res.append(func(atuple[n-1],atuple[0]))  
    return tuple(res)

def ReSubdivide(polygon,n=3):
       X = polygon 
       for k in range(n):
           X = Subdivide(X)
       return X 



def Spike(polygon, parameter=-0.2):
       return Subdivide(polygon,lambda x,y: parameter*1j*(y-x) + (x+y)/2)

   ## Random Polygons and Random Perturbations

def RandomPoint(lower_left, upper_right):
       """Constructs a random point (a complex number) inside a  a rectangle 
       given by its lower left and upper right corners."""
       x = random.uniform(lower_left.real, upper_right.real)
       y = random.uniform(lower_left.imag, upper_right.imag)
       return complex(x,y)

def RandomPolygon(lower_left, upper_right,n):
       """Constructs an n-sided polygon inside a rectangle given by its lower 
       left and upper right corners."""
       acc = [ ]
       for k in range(n):
           point = RandomPoint(lower_left, upper_right)
           acc.append(point)
       return tuple(acc)   

def Perturb(X,epsilon=0.05):
       "Returns a slightly perturbed version of polygon X and normalizes it"
       acc = []
       for x in X:
           point = complex(random.uniform(-epsilon,epsilon),
                           random.uniform(-epsilon,epsilon))
           acc.append(x+point)
       return Normalize(tuple(acc)) 

def RePerturb(polygon,n=2):
       X = polygon 
       for k in range(n):
           X = Perturb(X)
       return X 

### Convex hull of a polygon
  #Work in progress

### Minkowski sum of two convex polygons
  #Work in progress


### Normalizing polygons


def Normalize(X):
       """Displaces a polygon so that its center of mass lies at the origin
       and dilates it so its area equals 10"""
       CM = CenterMass(X)
       a = math.sqrt(area(X)/10)
       return tuple(map(lambda x: (x-CM)/a, X))


def features(polygon):
       "Feature space for polygons consisting of four basic invariants"
       length = Moment(0,0,0,polygon).real
       length_centermass = abs(Moment(1,0,0,polygon))
       length_rotational_moment = Moment(1,1,0,polygon).real
       area_rotational_moment = Moment(1,2,1,polygon).imag
       return (length, length_centermass, length_rotational_moment,
                    area_rotational_moment)

def features2(polygon):
       length = Moment(0,0,0,polygon).real
       length_centermass = abs(Moment(1,0,0,polygon))
       axial_symmetry = Moment(0,2,2,polygon).imag
       affine_invariant = affine(polygon)
       return (length,length_centermass,axial_symmetry,affine_invariant)

def clean(string,k):
       "This is used to format the features for the csv file"
       acc = ''
       for x in string:
           if x != ' ' and x != '(' and x != ')':
              acc = acc + x
       acc = acc + ',' + str(k)       
       return acc


### Small project: print out the features of three polygons directly to a .csv file.

#### IMPORTANT : Check the computations of moments.
#### Beware of normalization of area to 10 !!! What is the effect of that?

### TPNL invariant code (experimental)

def distances(polygon):
    "distances of vertices to center of mass of a normalized polygon"
    return tuple(map(abs,polygon))

def normals(polygon):
    "Returns the normals to the sides of a positively-oriented polygon"
    acc = []
    n = len(polygon)
    for k in range(n-1):
        acc.append(-1j*(polygon[k+1] - polygon[k])/abs(polygon[k+1] - polygon[k]))
    acc.append(-1j*(polygon[0] - polygon[n-1])/abs(polygon[0] - polygon[n-1]))
    return acc

def lengths(polygon):
    acc = []
    n = len(polygon)
    for k in range(n-1):
        acc.append(abs(polygon[k+1] - polygon[k]))
    acc.append(abs(polygon[0] - polygon[n-1]))
    return acc    

def TPNL(polygon): 
    acc = []
    n = len(polygon)
    for k in range(n-1):
        acc.append(polygon[k].conjugate() * (-1j) 
                   * (polygon[k+1] - polygon[k])/abs(polygon[k+1] - polygon[k]))
        acc.append(polygon[k+1].conjugate() * (-1j) 
                   * (polygon[k+1] - polygon[k])/abs(polygon[k+1] - polygon[k]))
    acc.append(polygon[n-1].conjugate() * (-1j) 
               * (polygon[0] - polygon[n-1])/abs(polygon[0] - polygon[n-1]))
    acc.append(polygon[0].conjugate() * (-1j) 
               * (polygon[0] - polygon[n-1])/abs(polygon[0] - polygon[n-1]))
    return acc  


## Trying the diagram with an ellipse.
t = np.linspace(0,2*np.pi,100)
x = 2*np.cos(t)
y = np.sin(t)
Elipse = tuple(map(complex,x,y))
xprime = -2*np.sin(t)
yprime = np.cos(t)
Elipseprime = tuple(map(complex, xprime,yprime))
Elipsenormals = tuple(map(lambda z: -1j*z/abs(z), Elipseprime))

def convex_hull(points):
    """Compute the convex hull of a set of points given as complex numbers."""
    # Convert complex numbers to tuples of floats for sorting
    points = [(p.real, p.imag) for p in points]

    # Sort the points lexographically (first by x, then by y)
    points = sorted(points)

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate and remove duplicates
    convex_hull_points = lower[:-1] + upper[:-1]

    # Convert back to complex numbers
    return [complex(p[0], p[1]) for p in convex_hull_points]

def cross(o, a, b):
    """Compute the cross product of vectors oa and ob."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def shape_to_complex(shape: np.ndarray) -> np.ndarray:
    """
    Convert a 2D shape array (N x 2) to complex numbers.

    Parameters
    ----------
    shape : np.ndarray
        Array of shape (N, 2), where each row is [x, y].

    Returns
    -------
    np.ndarray
        Array of complex numbers: x + iy for each point.
    """
    # Check input shape
    if shape.ndim != 2 or shape.shape[1] != 2:
        raise ValueError("Input shape must be a (N,2) array of coordinates.")
    
    # Convert to complex numbers
    return shape[:, 0] + 1j * shape[:, 1]

def remove_duplicate_vertices(X, tol=1e-12):
    """
    Removes consecutive duplicate (or near-duplicate) vertices
    from a closed polygon X.
    
    Parameters
    ----------
    X : iterable of complex
        Polygon vertices (closed or open)
    tol : float
        Distance tolerance for duplicates
    
    Returns
    -------
    tuple of complex
        Cleaned polygon
    """
    if len(X) < 2:
        return tuple(X)

    cleaned = [X[0]]

    for z in X[1:]:
        if abs(z - cleaned[-1]) > tol:
            cleaned.append(z)

    # If polygon is closed, check last vs first
    if len(cleaned) > 1 and abs(cleaned[0] - cleaned[-1]) < tol:
        cleaned.pop()

    return tuple(cleaned)

import numpy as np
import matplotlib.colors as mcolors

def generate_distinct_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [mcolors.hsv_to_rgb((h, 0.75, 0.9)) for h in hues]
    return colors

def plot_2D(X, labels):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    methods = {
    "PCA": PCA(n_components=2),
    "t-SNE": TSNE(n_components=2, random_state=42, perplexity=30),
    "UMAP": UMAP(n_components=2, random_state=42),
    "MDS": MDS(n_components=2, random_state=42),
    #"Isomap": Isomap(n_components=2),
    "Bouquet of Flowers": local_pca(X_scaled, labels)
}

    if len(set(labels)) > 1:
        methods["LDA"] = LDA(n_components=2)

    plt.figure(figsize=(18, 12))

    n_clusters = len(np.unique(labels))
    colors = generate_distinct_colors(n_clusters)
    for i, (name, model) in enumerate(methods.items(), 1):
        plt.subplot(2, 3, i)

        if name == "LDA":
            X_2d = model.fit_transform(X_scaled, labels)
        elif name == "Bouquet of Flowers":
            X_2d = model
        else:
            X_2d = model.fit_transform(X_scaled)

        for label_id in range(n_clusters):
            idx = labels == label_id
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], alpha=0.7, color=colors[label_id], label=f"Class {label_id}")

        plt.title(name)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.legend(loc="best", fontsize=8)

    plt.suptitle("Original Leaves Visualization", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def reparametrize_by_arc_length(curve, N, normalized = True):
    """ Reparameterize a 2D curve by its arc length.
    Parameters
    ----------
    curve : np.array
        An array of shape (nb_frame, 2) representing the 2D curve.
    N : int
        The number of points in the output curve.
    Returns
    -------
    newdelta : np.array
        An array of shape (N,) representing the new arc length parameterization.
    arc_length_parametrized_curve : np.array
        An array of shape (N, 2) representing the curve reparameterized by arc length.
    """
    dim = curve.shape[1]
    length = compute_length(curve)
    deltas = np.linalg.norm(np.diff(curve, axis=0), axis=1) + 1e-6
    deltas = np.insert(deltas, 0, 0.0)
    cumdelta = np.cumsum(deltas)
    if normalized:
        cumdelta = cumdelta/length
        newdelta = np.linspace(0, 1, N)
    else:
        newdelta = np.linspace(0, length, N) 
    #newdelta[-1] = cumdelta[-1] # ensure the last point matches exactly

    # Interpolate to uniform arc-length
    arc_length_parametrized_curve = np.zeros((N, dim))
    for i in range(dim):
        arc_length_parametrized_curve[:, i] = interp1d(cumdelta, curve[:, i], kind="linear", fill_value="extrapolate")(newdelta)
    return newdelta,arc_length_parametrized_curve

def compute_length(curve):
    """ Compute the length of a 2D curve.
    Parameters
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    length : float
        The length of the curve.
    """
    length = 0
    for i in range(curve.shape[0]-1):
        length += np.linalg.norm(curve[i+1,:]-curve[i,:])
    return length


def one_polygon_per_class(polygons, labels):
    """Return first polygon found for each class label."""
    selected = {}
    for poly, lab in zip(polygons, labels):
        selected.setdefault(lab, poly)
    return selected


def plot_polygon(ax, poly, *, color="k", lw=1, close=True):
    """Plot a complex-valued polygon on an axis."""
    pts = np.asarray(poly, dtype=complex)

    if close:
        pts = np.r_[pts, pts[0]]  # close polygon

    ax.plot(pts.real, pts.imag, color=color, linewidth=lw)
    ax.set_aspect("equal")
    ax.axis("off")


def local_pca(data, labels):
    """
    LPCA is a visualization procedure that computes 2D positions of labelled
    datapoints from a high-dimensional dataset by performing a global PCA
    followed by local PCAs in each class.
    """
    nb_of_samples, dim = data.shape
    nb_of_classes = len(np.unique(labels))
    
    # Global PCA
    global_PCA = PCA(n_components=None).fit(data)
    eigenvectors, eigenvalues = global_PCA.components_, global_PCA.explained_variance_
    cumulative_variance = np.cumsum(global_PCA.explained_variance_ratio_)
    estimated_global_dimension = np.where(cumulative_variance >= 0.99)[0][0]

    global_mean = np.mean(data, axis=0)
    data_in_global_PCA = global_PCA.transform(data)[:, :estimated_global_dimension]
    mean_data_in_global_PCA = np.mean(data_in_global_PCA, 0)

    dimension_per_class = np.zeros(nb_of_classes, dtype=int)
    dist_inter_class = []
    vectors = np.zeros((nb_of_classes, 2))
    centroid_positions = np.zeros((nb_of_classes, 2))
    rotation = []
    PCAs_per_class = []
    
    for i in range(nb_of_classes):
        class_data = data_in_global_PCA[labels == i]
        class_PCA = PCA(n_components=None).fit(class_data)
        PCAs_per_class.append(class_PCA)
        mean_class_data = np.mean(class_data, axis=0)
        
        cumulative_variance_class = np.cumsum(class_PCA.explained_variance_ratio_)
        dimension_per_class[i] = np.where(cumulative_variance_class >= 0.99)[0][0]
        
        # Distance between class mean and global mean
        dist_inter_class.append(np.linalg.norm(mean_data_in_global_PCA - mean_class_data))
        mean_displacement = mean_class_data[0:2] - mean_data_in_global_PCA[0:2]
        normalizing_term = np.linalg.norm(mean_displacement)  # ← FIXED
        vectors[i] = mean_displacement / normalizing_term
        centroid_positions[i] = dist_inter_class[i] * vectors[i]
        
        # Rotation angle
        projection_eigenvector_class = class_PCA.components_[0][0:2] / np.linalg.norm(class_PCA.components_[0][0:2])
        if projection_eigenvector_class[1] > 0:
            angle = np.arccos(projection_eigenvector_class[0])
        elif projection_eigenvector_class[1] < 0:
            angle = -np.arccos(projection_eigenvector_class[0])
        else:
            angle = 0
        rotation.append([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    global_positions = np.zeros((nb_of_samples, 2))
    
    for i in range(nb_of_samples):
        for j in range(nb_of_classes):
            if labels[i] == j:

                centered_point = data_in_global_PCA[i, :] - PCAs_per_class[j].mean_
                positions = PCAs_per_class[j].components_ @ data_in_global_PCA[i, :]
                
                # Scale by global eigenvalues
                positions[0] = positions[0] #* np.sqrt(eigenvalues[0])
                positions[1] = positions[1] #* np.sqrt(eigenvalues[1])
                
                # Rotate
                positions_2d = rotation[j] @ positions[0:2]
                
                # Translate to class centroid
                global_positions[i, 0] = positions_2d[0] + centroid_positions[j, 0]
                global_positions[i, 1] = positions_2d[1] + centroid_positions[j, 1]
    
    amb_space = min(nb_of_samples, dim)
    print(f"Dimension of ambient space: {amb_space},\nEstimated global dimension: {estimated_global_dimension},\n"
          f"The estimated dimensions per class are: {dimension_per_class}, \n"
          f"Upper bound local dimension = {max(dimension_per_class)}, \n"
          f"Centroids positions = {centroid_positions}")

    return global_positions


