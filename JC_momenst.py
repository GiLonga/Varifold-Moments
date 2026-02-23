### From Scheme to Python : moments.ss 


#Note 22/01/2026: the function Subdivide is not working as expected. Working on it ...
#Note 22/01/2026: bugged fixed: the problem was that in cutting and pasting the indentation
# had changed and that screwed the for loop.

import math
import random
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
    if np.isnan(multiplier):
         print("NAN encountered in MomentTripleSum with inputs:", p,q,r,z,w)
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


def moment_hu(p,q,X):
    """
    The Hu moment invariant of order p+q+1 formed with the moments m_{p,q,0} and m_{p,q+1,1}.
    """
    return Moment(p, q+1, 1, X) / (2j * (q+1))

def HuInvariants(X, normalize=False):
    """
    Compute Flusser rotation invariants up to order 3
    for polygon X (tuple of complex numbers).

    If normalize=True, returns similarity invariants
    (scale invariant).
    """

    # --- helper ---
    def c(p,q,X):
        return Moment(p, q+1, 1, X) / (2j * (q+1))

    Xc = Normalize(X)

    c11 = c(1,1,Xc)

    c20 = c(2,0,Xc)
    c02 = c(0,2,Xc)

    c30 = c(3,0,Xc)
    c03 = c(0,3,Xc)

    c21 = c(2,1,Xc)
    c12 = c(1,2,Xc)

    I1 = c11
    I2 = c20 * c02
    I3 = c30 * c03
    I4 = c21 * c12
    I5 = np.real(c30*c12**3)
    I6 = np.real(c20*c12**2)
    I7 = np.imag(c30*c12**3)

    invariants = {
        "I2": I2,
        "I3": I3,
        "I4": I4,
        "I5": I5,
        "I6": I6,
        "I7": I7
    }

    # --- 6) optional scale normalization ---
    if normalize:
        # c11 has scaling degree 2
        s2 = c11
        s3 = c11**2
        s4 = c11**3

        invariants = {
            "I2": I2 / s2,
            "I3": I3 / s3,
            "I4": I4 / s4,
            "I5": I5 / s4,
            "I6": I6 / s4
        }

    return invariants

def FlusserFullInvariants(X):
    """
    Compute Flusser's full set of independent rotation invariants
    (11 invariants) for a polygon X (tuple of complex numbers).
    All central moments computed once.
    """

    # --- helper: complex central moments ---
    def c(p,q,X):
        return Moment(p, q+1, 1, X) / (2j * (q+1))

    # --- 2) centralize polygon ---
    Xc = Normalize(X)

    # --- 3) compute all needed moments once ---
    c11 = c(1,1,Xc)
    c20 = c(2,0,Xc)
    c21 = c(2,1,Xc)
    c12 = c(1,2,Xc)
    c30 = c(3,0,Xc)
    c31 = c(3,1,Xc)
    c40 = c(4,0,Xc)
    c22 = c(2,2,Xc)

    # --- 4) compute the invariants ---
    invariants = {
        "F1": c11,
        "F2": c21 * c12,
        "F3": (c20 * (c12**2)).real,
        "F4": (c20 * (c12**2)).imag,
        "F5": (c30 * (c12**3)).real,
        "F6": (c30 * (c12**3)).imag,
        "F7": c22,
        "F8": (c31 * (c12**2)).real,
        "F9": (c31 * (c12**2)).imag,
        "F10": (c40 * (c12**4)).real,
        "F11": (c40 * (c12**4)).imag
    }

    return invariants

def flusser_invariants_norm(X):
    """
    Compute Flusser's full set of independent rotation invariants
    (11 invariants) for a polygon X (tuple of complex numbers).
    All central moments computed once.
    """

    # --- helper: complex central moments ---
    def c(p,q,X):
        return Moment(p, q+1, 1, X) / (2j * (q+1))

    # --- 2) centralize polygon ---
    Xc = Normalize(X)

    # --- 3) compute all needed moments once ---
    c11 = c(1,1,Xc)
    c20 = c(2,0,Xc)
    c21 = c(2,1,Xc)
    c12 = c(1,2,Xc)
    c30 = c(3,0,Xc)
    c31 = c(3,1,Xc)
    c40 = c(4,0,Xc)
    c22 = c(2,2,Xc)

    # --- 4) compute the invariants ---
    invariants = {
        "F1": c11,
        "F2": c21 * c12,
        "F3": (c20 * (c12**2)).real,
        "F4": (c20 * (c12**2)).imag,
        "F5": (c30 * (c12**3)).real,
        "F6": (c30 * (c12**3)).imag,
        "F7": c22,
        "F8": (c31 * (c12**2)).real,
        "F9": (c31 * (c12**2)).imag,
        "F10": (c40 * (c12**4)).real,
        "F11": (c40 * (c12**4)).imag
    }

    return invariants

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

import matplotlib.pyplot as plt

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
        # for loop ends here.
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

def RePerturb(polygon,n=2,epsilon=0.05):
       "Repeatedly Perturbs a polygon"
       X = polygon 
       for k in range(n):
           X = Perturb(X,epsilon)
       return X 

### Convex hull of a polygon
  #Work in progress

### Minkowski sum of two convex polygons
  #Work in progress


### Normalizing polygons


def Normalize(X, size = 10):
       """Displaces a polygon so that its center of mass lies at the origin
       and dilates it so its area equals 10"""
       CM = CenterMass(X)
       a = math.sqrt(area(X)/size)
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

####################################################################################
## These are the 80 invariants that we can concoct from the moments m_{p,q,r}
## with p,q,r non-negative integers and p + q + r < 4. There are a lot of dependencies:
## counting parameters in the moments themselves, I find that only 19 of these should be
## independent. Nevertheless the dependencies are not obvious and I went through the 
## (painful) trouble to list them all so that the machine can choose the ones it likes 
## best. 

## To avoid problems I had the fuction normalize the polygon first so that its area is 10
## and its center of mass is at the origin. The order of the invariants in the feature 
## vector is significant: first go the invariants formed with the moments m_{p,q,r}
## with p+q+r = 1, then the invariants formed with the moments with p+q+r = 2, then
## the invariants that mix moments with p+q+r = 1 and p+q+r = 2, then those formed
## with the moments with p+q+r=3, then mixing p+q+r=3 with p+q+r=2, then mixing p+q+r=3
## with p+q+r = 1.  Finally, we have the products of three moments. 

## The output of the Features is a tuple, but you can make it into an np array using
## np.asarray(Features(your polygon)). 

### A LOT OF INVARIANTS ###
def Features(polygon):
        "All moment invariants computed with moments of orders 0,1,2,and 3"
        X = Normalize(polygon)
        m000 = Moment(0,0,0,X)
        m100 = Moment(1,0,0,X)
        m200 = Moment(2,0,0,X)
        m110 = Moment(1,1,0,X)
        m002 = Moment(0,0,2,X)
        m300 = Moment(3,0,0,X)
        m003 = Moment(0,0,3,X)
        m210 = Moment(2,1,0,X)
        m102 = Moment(1,0,2,X)
        m012 = Moment(0,1,2,X)
        m300 = Moment(3,0,0,X)
        m003 = Moment(0,0,3,X)
        m210 = Moment(2,1,0,X)
        m102 = Moment(1,0,2,X)
        m012 = Moment(0,1,2,X)
        return (m000,
                abs(m100),
                abs(m200),
                abs(m002), 
                m200*m002.conjugate(),
                m110,
                m002.conjugate()*m100**2,
                m200.conjugate()*m100**2,
                abs(m300),
                abs(m003), 
                m300*m003.conjugate(), 
                abs(m210),
                m300*m210.conjugate()**3, 
                m003*m210.conjugate()**3, 
                abs(m102),
                m300*m102.conjugate(), 
                m003*m102.conjugate(), 
                m102*m210.conjugate()**3, 
                abs(m012),m300*m012.conjugate()**3,
                m003*m012.conjugate()**3, 
                m012*m210.conjugate(), 
                m102*m012.conjugate()**3,
                (m300**2)*m200.conjugate()**3,
                (m300**2)*m002.conjugate()**3,
                (m003**2)*m200.conjugate()**3,
                (m003**2)*m002.conjugate()**3,
                m200*m210.conjugate()**2,
                m002*m210.conjugate()**2, 
                (m102**2)*m200.conjugate()**3,
                (m102**2)*m002.conjugate()**3,
                m200*m012.conjugate()**2,
                m002*m012.conjugate()**2, 
                m300*m100.conjugate()**3,
                m003*m100.conjugate()**3, 
                m210*m100.conjugate(),
                m102*m100.conjugate()**3,
                m012*m100.conjugate(),
                m100*m210*m200.conjugate(),m100*m210*m002.conjugate(),
                m100*m210*m012.conjugate(),m100*m012*m002.conjugate(),
                m012*m210*m200.conjugate(),m012*m210*m002.conjugate(),
                (m100**2)*m210*m300.conjugate(),
                (m100**2)*m210*m003.conjugate(),
                (m100**2)*m210*m102.conjugate(),
                (m100**2)*m012*m300.conjugate(),
                (m100**2)*m012*m003.conjugate(),
                (m100**2)*m012*m102.conjugate(),
                (m210**2)*m100*m300.conjugate(),
                (m210**2)*m100*m003.conjugate(),
                (m210**2)*m100*m102.conjugate(),
                (m210**2)*m012*m300.conjugate(),
                (m210**2)*m012*m003.conjugate(),
                (m210**2)*m012*m102.conjugate(),
                (m012**2)*m100*m300.conjugate(),
                (m012**2)*m100*m003.conjugate(),
                (m012**2)*m100*m102.conjugate(),
                (m012**2)*m210*m300.conjugate(),
                (m012**2)*m210*m003.conjugate(),
                (m012**2)*m210*m102.conjugate(),
                m100*m200*m300.conjugate(),
                m100*m200*m003.conjugate(),
                m100*m200*m102.conjugate(),
                m100*m002*m300.conjugate(),
                m100*m002*m003.conjugate(), 
                m100*m002*m102.conjugate(),
                m210*m200*m300.conjugate(),
                m210*m200*m003.conjugate(),
                m210*m200*m102.conjugate(),
                m210*m002*m300.conjugate(),
                m210*m002*m003.conjugate(), 
                m210*m002*m102.conjugate(),
                m012*m200*m300.conjugate(),
                m012*m200*m003.conjugate(),
                m012*m200*m102.conjugate(),
                m012*m002*m300.conjugate(),
                m012*m002*m003.conjugate(), 
                m012*m002*m102.conjugate())

### ACTUALLY OLD FEATURES ###
def features_new(polygon):
    """
    All moment invariants computed with moments of orders 0, 1, 2, and 3.
    Returns a dictionary with descriptive keys for clarity.
    """
    X = Normalize(polygon)

    m000 = Moment(0,0,0,X)
    m100 = Moment(1,0,0,X)
    m200 = Moment(2,0,0,X)
    m110 = Moment(1,1,0,X)
    m002 = Moment(0,0,2,X)
    m300 = Moment(3,0,0,X)
    m003 = Moment(0,0,3,X)
    m210 = Moment(2,1,0,X)
    m102 = Moment(1,0,2,X)
    m012 = Moment(0,1,2,X)

    features_list = [
        m000, abs(m100), abs(m200), abs(m002), m200*m002.conjugate(),
        m110, m002.conjugate()*m100**2, m200.conjugate()*m100**2,
        abs(m300), abs(m003), m300*m003.conjugate(), abs(m210),
        m300*m210.conjugate()**3, m003*m210.conjugate()**3, abs(m102),
        m300*m102.conjugate(), m003*m102.conjugate(), 
        m102*m210.conjugate()**3, abs(m012), m300*m012.conjugate()**3,
        m003*m012.conjugate()**3, m012*m210.conjugate(), 
        m102*m012.conjugate()**3, (m300**2)*m200.conjugate()**3,
        (m300**2)*m002.conjugate()**3, (m003**2)*m200.conjugate()**3,
        (m003**2)*m002.conjugate()**3, m200*m210.conjugate()**2,
        m002*m210.conjugate()**2, (m102**2)*m200.conjugate()**3,
        (m102**2)*m002.conjugate()**3, m200*m012.conjugate()**2,
        m002*m012.conjugate()**2, m300*m100.conjugate()**3,
        m003*m100.conjugate()**3, m210*m100.conjugate(),
        m102*m100.conjugate()**3, m012*m100.conjugate(),
        m100*m210*m200.conjugate(), m100*m210*m002.conjugate(),
        m100*m210*m012.conjugate(), m100*m012*m002.conjugate(),
        m012*m210*m200.conjugate(), m012*m210*m002.conjugate(),
        (m100**2)*m210*m300.conjugate(),
        (m100**2)*m210*m003.conjugate(),
        (m100**2)*m210*m102.conjugate(),
        (m100**2)*m012*m300.conjugate(),
        (m100**2)*m012*m003.conjugate(),
        (m100**2)*m012*m102.conjugate(),
        (m210**2)*m100*m300.conjugate(),
        (m210**2)*m100*m003.conjugate(),
        (m210**2)*m100*m102.conjugate(),
        (m210**2)*m012*m300.conjugate(),
        (m210**2)*m012*m003.conjugate(),
        (m210**2)*m012*m102.conjugate(),
        (m012**2)*m100*m300.conjugate(),
        (m012**2)*m100*m003.conjugate(),
        (m012**2)*m100*m102.conjugate(),
        (m012**2)*m210*m300.conjugate(),
        (m012**2)*m210*m003.conjugate(),
        (m012**2)*m210*m102.conjugate(),
        m100*m200*m300.conjugate(), m100*m200*m003.conjugate(),
        m100*m200*m102.conjugate(), m100*m002*m300.conjugate(),
        m100*m002*m003.conjugate(), m100*m002*m102.conjugate(),
        m210*m200*m300.conjugate(), m210*m200*m003.conjugate(),
        m210*m200*m102.conjugate(), m210*m002*m300.conjugate(),
        m210*m002*m003.conjugate(), m210*m002*m102.conjugate(),
        m012*m200*m300.conjugate(), m012*m200*m003.conjugate(),
        m012*m200*m102.conjugate(), m012*m002*m300.conjugate(),
        m012*m002*m003.conjugate(), m012*m002*m102.conjugate()
    ]

    # create dictionary with feature_1, feature_2, ..., feature_80
    features_dict = {f"feature_{idx+1}": val for idx, val in enumerate(features_list)}

    return features_dict



#### OFFICIAL INVARIANTSSSSS ###

def vtld(polygon, Norm = True, moment_dict = None):
    "Variance of the distribution of distances from tangent lines to cm"
    
    if moment_dict is None:
        moment_dict = {}

    if Norm:
        X = Normalize(polygon, size = 1)
    else:
        X = polygon
    m000 = get_moment(moment_dict, X, 0,0,0).real
    m = 2*area(X)/m000
    m022 = get_moment(moment_dict, X, 0,2,2).real
    m110 = get_moment(moment_dict, X, 1,1,0).real
    return (m110 - m022)/(2*m000) - m**2

def msdb(polygon, Norm = True, moment_dict = None):
    "Mean of square distance from boundary to center of mass"
    
    if moment_dict is None:
        moment_dict = {}
    if Norm:
        X = Normalize(polygon, size = 1)
    else:
        X = polygon
    return get_moment(moment_dict, X, 1,1,0)/get_moment(moment_dict, X, 0,0,0)

def vsdb(polygon, Norm = True , moment_dict = None):
    "Variance of square distance from boundary to center of mass"

    if moment_dict is None:
        moment_dict = {}

    if Norm:
        X = Normalize(polygon, size = 1)
    else:
        X = polygon
    m000 = get_moment(moment_dict, X, 0,0,0)
    m110 = get_moment(moment_dict, X, 1,1,0)
    m220 = get_moment(moment_dict, X, 2,2,0)
    return m220/m000 - (m110/m000)**2 

def dist2cm(polygon, Norm = True , moment_dict = None):
    "TO DO "
 
    if moment_dict is None:
        moment_dict = {}

    if Norm:
        X = Normalize(polygon, size = 1)
    else:
        X = polygon

    m011 = get_moment(moment_dict, X, 0,1,1).imag
    m121 = get_moment(moment_dict, X, 1,2,1).imag
    m1 = m121/(2*m011)
    m231 = get_moment(moment_dict, X, 2,3,1).imag
    m2 = m231/(3*m011)
    m2c = m2 - m1**2
    m341 = get_moment(moment_dict, X, 3,4,1).imag
    m3 = m341/(4*m011)
    m3c = m3 - 3*m1*m2 + 2*m1**3
    skewness = m3c/math.sqrt(m2c)**3
    m451 = get_moment(moment_dict, X, 4,5,1).imag
    m4 = m451/(5*m011) 
    m4c = m4 - 4*m1*m3 + 6*m2*m1**2 - 3*m1**4
    kurtosis = m4c/m2c**2

    return (m1,m2c,skewness,kurtosis)

def skewsdb(polygon, Norm = True, moment_dict = None):
    "Skewness of square distance from boundary to center of mass" 


    if moment_dict is None:
        moment_dict = {}
    
    if Norm:
        X = Normalize(polygon, size = 1)
    else:
        X = polygon
    
    m000 = get_moment(moment_dict, X, 0,0,0)
    m110 = get_moment(moment_dict, X, 1,1,0)
    m220 = get_moment(moment_dict, X, 2,2,0)
    m330 = get_moment(moment_dict, X, 3,3,0)
    m3   = (m330/m000 - 3*m220*m110/m000**2 + 2*(m110/m000)**3).real 
    m2   = vsdb(X, Norm=False, moment_dict = moment_dict).real  
    return m3/math.sqrt(m2**3)

def kurtosissdb(polygon, Norm = True, moment_dict = None):
    "Kurtosis of square distance from boundary to center of mass" 

    if moment_dict is None:
        moment_dict = {}
    
    if Norm:
        X = Normalize(polygon, size = 1)
    else:
        X = polygon
    
    m000 = get_moment(moment_dict, X, 0,0,0).real
    m110 = get_moment(moment_dict, X, 1,1,0).real
    m220 = get_moment(moment_dict, X, 1,1,0).real
    m330 = get_moment(moment_dict, X, 3,3,0).real
    m440 = get_moment(moment_dict, X, 4,4,0).real
    m2   = vsdb(X, Norm = False, moment_dict = moment_dict).real
    m4   = m440/m000 - 4*m330*m110/m000**2 + 6*(m220*m110**2)/m000*3 - 3*(m110/m000)**4
    return m4/m2**2 

def point2line(polygon, Norm = True, moment_dict = None):
    
    if moment_dict is None:
        moment_dict = {}

    if Norm:
        X = Normalize(polygon, size = 1)
    else:
        X = polygon
    m000 = get_moment(moment_dict, X, 0,0,0).real
    m1 = -2*area(X)/m000
    m100 = get_moment(moment_dict, X, 1,0,0)
    m110 = get_moment(moment_dict, X, 1,1,0).real
    m020 = get_moment(moment_dict, X, 0,2,0)
    m002 = get_moment(moment_dict, X, 0,0,2)
    m012 = get_moment(moment_dict, X, 0,1,2)
    m022 = get_moment(moment_dict, X, 0,2,2)
    a = (m020*m002).real
    b = (m012*m100.conjugate()).real
    m2 = ((-1/2)*(a - 2*b + m000*m022.real) + m000*m110 - abs(m100)**2)/m000**2
    m2c = m2 - m1**2
    m030 = get_moment(moment_dict, X, 0,3,0)
    m003 = get_moment(moment_dict, X, 0,0,3)
    m013 = get_moment(moment_dict, X, 0,1,3)
    m023 = get_moment(moment_dict, X, 0,2,3)
    m033 = get_moment(moment_dict, X, 0,3,3)
    m121 = get_moment(moment_dict, X, 1,2,1).imag
    x = (m030*m003 - 3*m020*m013 + 3*m023*m100.conjugate() - m033).imag
    y = -3*m110*area(X) - 3*m000*m121.imag/4
    m3 = (-x/4 + y)/m000**2
    m3c = m3 - 3*m1*m2 + 2*m1**2
    return (m2c, m3c/math.sqrt(m2c)**3)

def borderdets(polygon, Norm = True, moment_dict = None):
    """ TO DO """

    if moment_dict is None:
        moment_dict = {}

    if Norm:
        X = Normalize(polygon, size = 1)
    else:
        X = polygon

    m000 = get_moment(moment_dict, X, 0,0,0).real
    m002 = get_moment(moment_dict, X, 0,0,2)
    m2 = (m000**2 - abs(m002)**2)/(2*m000**2)
    m004 = get_moment(moment_dict, X, 0,0,4)
    m4 = (abs(m004)**2 - 4*abs(m002)**2 + 3*m000**2)/(8*m000**2)
    return(m2,m4/m2**2)
########################################################################
def features_CP(polygon):
    "Testing the idea of moment invariants as points in R^k x CP^n"
    X = Normalize(polygon)
    m000 = Moment(0,0,0,X).real
    m100 = Moment(1,0,0,X)
    m200 = Moment(2,0,0,X)
    m110 = Moment(1,1,0,X).real
    m002 = Moment(0,0,2,X)
    m210 = Moment(2,1,0,X)
    m012 = Moment(0,1,2,X)
    m220 = Moment(2,2,0,X).real
    m121 = Moment(1,2,1,X).imag
    m022 = Moment(0,2,2,X)
    m310 = Moment(3,1,0,X)
    m211 = Moment(2,1,1,X)
    m112 = Moment(1,1,2,X)
    m013 = Moment(0,1,3,X)
    m003 = Moment(0,0,3,X)
    m113 = Moment(1,1,3,X)
    m014 = Moment(0,1,4,X)
    m223 = Moment(2,2,3,X)
    m212 = Moment(2,1,2,X)
    m311 = Moment(3,1,1,X)
    m300 = Moment(3,0,0,X)
    m410 = Moment(4,1,0,X)
    m322 = Moment(3,2,2,X)
    m300 = Moment(3,0,0,X)
    m333 = Moment(3,3,3,X)

    features_list = [m000,
                        m110,
                        m220,
                        m121,
                        m022,
                        m100**2,
                        m210**2,
                        m012**2,
                        m200,
                        m002,
                        m310,
                        m211,
                        m112,
                        m013,

                        m003,
                        m113,
                        m014,
                        m223,
                        m212,
                        m311,
                        m300,
                        m410,
                        m322,
                        m300,
                        m333,
                        ] 
    for i, moment in enumerate(features_list):
        if i>0 and i<5:
            features_list[i] = moment/m000
        elif i>4 and i<14:
            features_list[i] = moment/m211
        elif i > 13:
            features_list[i] = moment/m333
             
    features_dict = {f"feature_{idx+1}": val for idx, val in enumerate(features_list)}
    return features_dict

def Normalized_Features(polygon):
    "Testing the idea of moment invariants as points in R^k x CP^n"
    X = Normalize(polygon)
    m000 = Moment(0,0,0,X).real
    m100 = Moment(1,0,0,X)/m000
    m200 = Moment(2,0,0,X)/m000
    m110 = Moment(1,1,0,X).real/m000
    m220 = Moment(2,2,0,X).real/m000
    m121 = Moment(1,2,1,X).imag
    m022real = Moment(0,2,2,X).real
    m022imag = Moment(0,2,2,X).imag
    m002 = Moment(0,0,2,X)
    m210 = Moment(2,1,0,X)/m000
    m012 = Moment(0,1,2,X)
    m310 = Moment(3,1,0,X)/m000
    m211 = Moment(2,1,1,X)
    m112 = Moment(1,1,2,X)
    m013 = Moment(0,1,3,X)
    features_list = [m110,
                        m220,
                        m121,
                        m022real,
                        m022imag,
                        m100**2,
                        m210**2,
                        m012**2,
                        m200,
                        m002,
                        m310,
                        m211,
                        m112,
                        m013]

    features_dict = {f"feature_{idx+1}": val for idx, val in enumerate(features_list)}
    
    return features_dict
    
def compute_moments(X, moment_list):
    cache = {}
    for (p,q,r) in moment_list:
        cache[(p,q,r)] = Moment(p,q,r,X)
    return cache

def get_moment(moment_dict, X, p, q, r):
    key = (p, q, r)
    if key in moment_dict:
        return moment_dict[key]
    print(f"The moment m_{{{p},{q},{r}}} is not in the cache. Computing it now.")
    print(moment_dict.keys())
    val = Moment(p, q, r, X)
    moment_dict[key] = val
    return val
def features_13(polygon):

    polygon_ = Normalize(polygon, size = 1)

    moments_list = [(0,0,0), (0,2,2), (1,1,0), (0,1,1), (2,2,0), (3,3,0), (4,4,0), (1,0,0), (0,2,0), (0,0,2), (0,1,2), (0,3,0), (0,0,3), (0,1,3), (0,2,3), (0,3,3), (1,2,1), (0,0,4), (3,1,1),]
    cache_moments = compute_moments(polygon_, moments_list)

    I1 = msdb(polygon_, False, moment_dict = cache_moments)
    I2 = vsdb(polygon_, False, moment_dict = cache_moments)
    I3 = skewsdb(polygon_, False, moment_dict = cache_moments)
    I4 = kurtosissdb(polygon_, False, moment_dict = cache_moments)

    m100 = get_moment(cache_moments, polygon_, 1,0,0)
    m000 = get_moment(cache_moments, polygon_, 0,0,0)
    I5 = abs(m100/m000)

    m311 = get_moment(cache_moments, polygon_, 3,1,1)
    m011 = get_moment(cache_moments, polygon_, 0,1,1)
    I6 = abs(m311/m011)

    m002 = get_moment(cache_moments, polygon_, 0,0,2)
    m003 = get_moment(cache_moments, polygon_, 0,0,3)

    I7= np.imag(m002)/m000
    I8 = np.real(m003)/m000

    I9 = vtld(polygon_, False, moment_dict = cache_moments)
    I10 = point2line(polygon_, False, moment_dict = cache_moments)[0]
    I11 = borderdets(polygon_, False,moment_dict = cache_moments)[0]

    invariants = {
        "JC1": I1,
        "JC2": I2,
        "JC3": I3,
        "JC4": I4,
        "JC5": I5,
        "JC6": I6,
        "JC7": I7,
        "JC8": I8,
        "JC9": I9,
        "JC10": I10,
        "JC11": I11
    }

    return invariants

def thirteen(x):
    "The thirteen invariants"

    X = Normalize(X, size = 1)

    moments_list = [(0,0,0), (0,2,2), (1,1,0), (0,1,1), (2,2,0), (3,3,0), (4,4,0), (1,0,0), (0,2,0), (0,0,2), (0,1,2), (0,3,0), (0,0,3), (0,1,3), (0,2,3), (0,3,3), (1,2,1), (0,0,4), (3,1,1),]
    cache_moments = compute_moments(X, moments_list)

    m000 = get_moment(cache_moments, X, 0,0,0).real
    m100 = get_moment(cache_moments, X, 1,0,0)
    m011 = get_moment(cache_moments, X, 0,1,1).imag
    m311 = get_moment(cache_moments, X, 3,1,1)
    m022 = get_moment(cache_moments, X, 0,2,2)
    m033 = get_moment(cache_moments, X, 0,3,3)
    I1, I2, I3, I4 = dist2cm(X, Norm = False, moment_dict= cache_moments)
    return (I1, I2, I3, I4,
            linear_inv(x)[0], linear_inv(x)[1],abs(m100/m000),abs(m311/m011),
            m022.imag/m000,m033.real/m000,borderdets(x)[0],vtld(x),point2line(x)[0])
### I completely forgot why I was doing this !
def clean(string,k):
    "This is used to format the features for the csv file"
    acc = ''
    for x in string:
        if x != ' ' and x != '(' and x != ')':
            acc = acc + x
    acc = acc + ',' + str(k)       
    return acc

def linear_inv(polygon):
    """TO DO, THIS IS JUST A STUBB """
    return (Moment(1,0,0,polygon), Moment(0,1,0,polygon))
### Small project: print out the features of three polygons directly to a .csv file.

#### IMPORTANT : Check the computations of moments.
#### Beware of normalization of area to 10 !!! What is the effect of that?

### TPNL invariant code (experimental). I forgot why or how I came up with this. 

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
