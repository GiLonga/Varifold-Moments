from JC_functions import *

def Triangle():
    return regular_gon(3)

def Square():
    return regular_gon(4)

def Pentagon():
    return regular_gon(5)

def Hexagon():
    return regular_gon(6)

def Heptagon():
    return regular_gon(7)

def Octagon():
    return regular_gon(8)

def Nonagon():
    return regular_gon(9)

def Circle(n=64):
    """Approximate a circle with n vertices"""
    return regular_gon(n)

def Star(n=5, spike=-0.5):
    """Regular star polygon"""
    return Spike(regular_gon(n), parameter=spike)

def RandomTransform(polygon,
                    scale_range=(0.5, 2.0),
                    shift_range=(-4, 4)):
    # random rotation
    angle = random.uniform(0, 2*math.pi)
    P = Rotate(angle, polygon)

    # random scaling (uniform)
    s = random.uniform(*scale_range)
    P = tuple(s * z for z in P)

    # random translation
    tx = random.uniform(*shift_range)
    ty = random.uniform(*shift_range)
    shift = complex(tx, ty)
    P = tuple(z + shift for z in P)

    return P

SHAPES = [
    ("Triangle", Triangle),
    ("Square", Square),
    ("Pentagon", Pentagon),
    ("Hexagon", Hexagon),
    ("Heptagon", Heptagon),
    ("Octagon", Octagon),
    ("Nonagon", Nonagon),
    ("Circle", Circle),
    ("Star", Star),
]

def GenerateDataset(samples_per_class=100):
    dataset = []

    for label, (name, shape_fn) in enumerate(SHAPES):
        for _ in range(samples_per_class):
            base = shape_fn()
            poly = RandomTransform(base)
            poly = Normalize(poly)  # optional but recommended
            dataset.append({
                "polygon": poly,
                "label": label,
                "name": name
            })

    return dataset

def GenerateFeatureDataset(samples_per_class=100):
    X = []
    y = []

    for label, (name, shape_fn) in enumerate(SHAPES):
        for _ in range(samples_per_class):
            poly = RandomTransform(shape_fn())
            poly = Normalize(poly)
            X.append(features(poly))
            y.append(label)

    return np.array(X), np.array(y)

def edge_lengths(polygon):
    n = len(polygon)
    return [abs(polygon[(k+1)%n] - polygon[k]) for k in range(n)]

def total_length(polygon):
    return sum(edge_lengths(polygon))

def ResampleArcLength(polygon, N=500):
    n = len(polygon)
    lengths = edge_lengths(polygon)
    L = sum(lengths)

    # cumulative arc-length
    cum = [0]
    for l in lengths:
        cum.append(cum[-1] + l)

    new_points = []
    step = L / N
    target = 0
    edge = 0

    for _ in range(N):
        while target > cum[edge+1]:
            edge += 1

        z0 = polygon[edge]
        z1 = polygon[(edge+1)%n]
        t = (target - cum[edge]) / lengths[edge]
        new_points.append((1-t)*z0 + t*z1)

        target += step

    return tuple(new_points)

def Shape500(shape_fn, N=500):
    base = shape_fn()
    smooth = ResampleArcLength(base, N)
    return Normalize(smooth)

def RandomShape500(shape_fn, N=500):
    poly = Shape500(shape_fn, N)
    poly = Rotate(random.uniform(0,2*math.pi), poly)

    s = random.uniform(0.7, 1.8)
    poly = tuple(s*z for z in poly)

    shift = complex(random.uniform(-4,4), random.uniform(-4,4))
    poly = tuple(z + shift for z in poly)

    return poly

def OrientCCW(polygon):
    """
    Ensures polygon is counter-clockwise oriented.
    If polygon is clockwise, it is reversed.
    """
    if area(polygon) < 0:
        return tuple(reversed(polygon))
    return polygon

def GenerateShapeArray(
        samples_per_class=100,
        points_per_shape=500,
        perturbation_strength=0.02):

    polygons = []
    labels = []
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    for label, (name, shape_fn) in enumerate(SHAPES):
        for _ in range(samples_per_class):

            poly = ResampleArcLength(shape_fn(), points_per_shape)
            poly = Rotate(random.uniform(0, 2 * math.pi), poly)

            scale = random.uniform(0.7, 1.8)
            poly = tuple(scale * z for z in poly)

            # Perturbation:
            poly = tuple(
                z + complex(
                    random.uniform(-perturbation_strength, perturbation_strength),
                    random.uniform(-perturbation_strength, perturbation_strength)
                )
                for z in poly
            )

            poly = ResampleArcLength(poly, points_per_shape)
            poly = OrientCCW(poly)

            #I NORMALIZE AGAIN JUST FOR SAFETY
            poly = Normalize(poly)

            polygons.append(poly)
            labels.append(label)

    return np.array(polygons, dtype=object), np.array(labels)
