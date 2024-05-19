def central_diff_x(f):
    diff = np.zeros_like(f)
    diff [1:-1,1:-1] = (f[1:-1,2:]-f[1:-1,0:-2])/(2*dx)
    return diff
def central_diff_y(f):
    diff = np.zeros_like(f)
    diff [1:-1,1:-1] = (f[2:,1:-1]-f[0:-2,1:-1])/(2*dy)
    return diff
def laplace(f):
    diff = np.zeros_like(f)
    diff [1:-1,1:-1] = (f[1:-1,0:-2]+f[0:-2,1:-1]-4*f[1:-1,1:-1]+f[1:-1,2:]+f[2:,1:-1])/(dy**2)
    return diff