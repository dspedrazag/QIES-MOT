import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

mu0 = 4 * np.pi * 1e-7

def BxIntegrand(phi, x, y, z, R, I, cx, cy, cz):
    x -= cx
    y -= cy
    z -= cz
    return ((mu0 * I) / (4 * np.pi)) * ((R * z * np.cos(phi)) / (((x - R * np.cos(phi))**2 + (y - R * np.sin(phi))**2 + z**2)**(3/2)))

def ByIntegrand(phi, x, y, z, R, I, cx, cy, cz):
    x -= cx
    y -= cy
    z -= cz
    return ((mu0 * I) / (4 * np.pi)) * ((R * z * np.sin(phi)) / (((x - R * np.cos(phi))**2 + (y - R * np.sin(phi))**2 + z**2)**(3/2)))

def BzIntegrand(phi, x, y, z, R, I, cx, cy, cz):
    x -= cx
    y -= cy
    z -= cz
    return ((mu0 * I) / (4 * np.pi)) * ((R**2 - R * (x * np.cos(phi) + y * np.sin(phi))) / (((x - R * np.cos(phi))**2 + (y - R * np.sin(phi))**2 + z**2)**(3/2)))

def oneLoopField(x, y, z, R, I, cx, cy, cz):
    #Bx  = quad(BxIntegrand, 0, 2 * np.pi, args=(x, y, z, R, I, cx, cy, cz))[0]
    #By  = quad(ByIntegrand, 0, 2 * np.pi, args=(x, y, z, R, I, cx, cy, cz))[0]
    #Bz = quad(BzIntegrand, 0, 2 * np.pi, args=(x, y, z, R, I, cx, cy, cz))[0]
    Bx = 0
    By = 0
    Bz = mu0 * I * R**2 / (2*(R**2+((x-cx)**2 + (y-cy)**2 + (z-cz)**2))**(3/2))

    Bx *= 1e4
    By *= 1e4
    Bz *= 1e4
    
    return np.array([Bx, By, Bz])


def antiHelmholtzField(x, y, z, R, I, d, num_layers, num_loops, Rwire):

    Btot = np.array([0, 0, 0], dtype=float)

    for i in range (num_layers):
        for j in range (num_loops):
            Bupper = oneLoopField(x, y, z, R + 2*Rwire*j, I, 0, 0, d + 2*Rwire*i)
            Blower = oneLoopField(x, y, z, R + 2*Rwire*j, -I, 0, 0, -d - 2*Rwire*i)
            Btot += Bupper
            Btot += Blower

    return Btot


def fieldVScurrent(x, y, z, R, d, num_layers, num_loops, Rwire):
    B_list = []
    I_list = np.linspace(0, 10, 1000)
    for I in I_list:
        B = antiHelmholtzField(x, y, z, R, I, d, num_layers, num_loops, Rwire)
        B_list.append(np.linalg.norm(B))

    grad = np.gradient(B_list, I_list[1] - I_list[0])
    
    plt.figure()
    plt.plot(I_list, B_list, label="B")
    plt.plot(I_list, grad, label="Gradient")
    plt.xlabel("I")
    plt.title("B field and gradient VS I")
    plt.legend()
    plt.grid(True)
    plt.show()


def fieldVSloopRadius(x, y, z, I, d, num_layers, num_loops, Rwire):
    B_list = []
    R_list = np.linspace(0, 1, 1000)
    for R in R_list:
        B = antiHelmholtzField(x, y, z, R, I, d, num_layers, num_loops, Rwire)
        B_list.append(np.linalg.norm(B))

    grad = np.gradient(B_list, R_list[1] - R_list[0])
    
    plt.figure()
    plt.plot(R_list, B_list, label="B")
    plt.plot(R_list, grad, label="Gradient")
    plt.xlabel("R")
    plt.title("B field and gradient VS loop radius")
    plt.legend()
    plt.grid(True)
    plt.show()

def fieldVSwireRadius(x, y, z, R, I, d, num_layers, num_loops):
    B_list = []
    R_list = np.linspace(0, .001, 1000)
    for Rwire in R_list:
        B = antiHelmholtzField(x, y, z, R, I, d, num_layers, num_loops, Rwire)
        B_list.append(B[2])

    grad = np.gradient(B_list, R_list[1] - R_list[0])
    for i in range (len(grad)):
        grad[i] = grad[i]/100
    
    plt.figure()
    plt.plot(R_list, B_list, label="B")
    #plt.plot(R_list, grad, label="Gradient")
    plt.xlabel("R")
    plt.title("B field VS wire radius")
    plt.legend()
    plt.grid(True)
    plt.show()

def fieldVSdistance(x, y, z, R, I, num_layers, num_loops, Rwire):
    B_list = []
    D_list = np.linspace(0, 0.5, 1000)
    for d in D_list:
        B = antiHelmholtzField(x, y, z, R, I, d, num_layers, num_loops, Rwire)
        B_list.append(np.linalg.norm(B))

    grad = np.gradient(B_list, D_list[1] - D_list[0])
    
    plt.figure()
    plt.plot(D_list, B_list, label="B")
    plt.plot(D_list, grad, label="Gradient")
    plt.xlabel("d")
    plt.title("B field and gradient VS coil separation")
    plt.legend()
    plt.grid(True)
    plt.show()

def plotZfield(x, y, R, I, d, num_layers, num_loops, Rwire, plotGrad):
    B_list = []
    Bz_list = []
    Z_list = np.linspace(-.01, .01, 1000)
    for z in Z_list:
        B = antiHelmholtzField(x, y, z, R, I, d, num_layers, num_loops, Rwire)
        if (z < 0):
            B_list.append(0-np.linalg.norm(B))
        else:
            B_list.append(np.linalg.norm(B))
        Bz_list.append(B[2])
    return Bz_list
    grad = np.gradient(B_list, Z_list[1] - Z_list[0])
    zGrad = np.gradient(Bz_list, Z_list[1] - Z_list[0])

    for i in range (len(zGrad)):
        zGrad[i] = zGrad[i]/100
        grad[i] = grad[i]/100
    #return Bz_list
    print(zGrad[500])
    print(Bz_list[500] - Bz_list[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(Z_list, Bz_list, label="Bz")
    axes[0].set_xlabel("z")
    axes[0].set_title("B field VS z")

    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(Z_list, zGrad, label="Gradient")
    axes[1].set_xlabel("z")
    plt.title("Gradient VS z")
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotXfield(y, z, R, I, d, num_layers, num_loops, Rwire, plotGrad):
    B_list = []
    Bx_list = []
    X_list = np.linspace(-.01, .01, 1000)
    for x in X_list:
        B = antiHelmholtzField(x, y, z, R, I, d, num_layers, num_loops, Rwire)
        if (x < 0):
            B_list.append(0-np.linalg.norm(B))
        else:
            B_list.append(np.linalg.norm(B))
        Bx_list.append(B[0])
    return Bx_list

    grad = np.gradient(B_list, X_list[1] - X_list[0])
    xGrad = np.gradient(Bx_list, X_list[1] - X_list[0])

    for i in range (len(xGrad)):
        xGrad[i] = xGrad[i]/100
        grad[i] = grad[i]/100
    print(Bx_list)
    
    plt.figure()
    #plt.plot(X_list, B_list, label="B")
    #plt.plot(X_list, Bx_list, label="Bx")
    if (plotGrad):
        plt.plot(X_list, grad, label="dB/dx")
        plt.plot(X_list, xGrad, label="Gradient")
    plt.xlabel("x")
    plt.title("B field and gradient VS x")
    plt.legend()
    plt.grid(True)
    plt.show()

def plotYfield(x, z, R, I, d, num_layers, num_loops, Rwire, plotGrad):
    B_list = []
    Y_list = np.linspace(-1, 1, 1000)
    By_list = []
    for y in Y_list:
        B = antiHelmholtzField(x, y, z, R, I, d, num_layers, num_loops, Rwire)
        if (y < 0):
            B_list.append(0 - np.linalg.norm(B))
        else:
            B_list.append(np.linalg.norm(B))
        By_list.append(B[1])

    grad = np.gradient(B_list, Y_list[1] - Y_list[0])
    yGrad = np.gradient(By_list, Y_list[1] - Y_list[0])

    plt.figure()
    plt.plot(Y_list, B_list, label="B")
    plt.plot(Y_list, By_list, label="By")
    if (plotGrad):
        plt.plot(Y_list, grad, label="dB/dy")
        plt.plot(Y_list, yGrad, label="Gradient")
    plt.xlabel("y")
    plt.title("B field and gradient VS y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    x = 0
    y = 0
    z = 0
    R = ((4.225+3/16/2)*25.4)*1e-3 #radius of innermost loop
    I = 6.77 #amps
    d = (((4.88+0.3*2+3/16)*25.4)/2)*1e-3 #d/2 to closest layer, center of wire
    num_layers = 6 #stacked vertically
    num_loops = 5
    Rwire = (((3/16+0.01)*25.4)/2)*1e-3

    x = 0
    y = 0
    z = 0
    R = 17.8562*1e-3 #radius of innermost loop
    I = 6.77 #amps
    d = (40.2844/2)*1e-3 #d/2 to closest layer, center of wire
    num_layers = 2 #stacked vertically
    num_loops = 8
    Rwire = 0.635*1e-3

    """
    I = optI
    R = optR
    d = optD
    Rwire = optRwire
    """
    
    B = antiHelmholtzField(x, y, z, R, I, d, num_layers, num_loops, Rwire)
    print(f"Point: ({x}, {y}, {z}). B: {B}. Magnitude: {np.linalg.norm(B)}")
    plotGrad = True

    if (True):
        plotZfield(x, y, R, I, d, num_layers, num_loops, Rwire, plotGrad)
        plotXfield(y, z, R, I, d, num_layers, num_loops, Rwire, plotGrad)
        #plotYfield(x, z, R, I, d, num_layers, num_loops, Rwire, plotGrad)

    if (False):
        fieldVSdistance(x, y, z, R, I, num_layers, num_loops, Rwire)
        fieldVScurrent(x, y, z, R, d, num_layers, num_loops, Rwire)
        fieldVSloopRadius(x, y, z, I, d, num_layers, num_loops, Rwire)
        fieldVSwireRadius(x, y, z, R, I, d, num_layers, num_loops)
        
if __name__ == "__main__":
    x = 0
    y = 0
    z = 0.0001
    R = 0.1778 #radius of innermost loop
    I = 165 #amps
    d = 0.03730625
    R = 0.0889
    num_layers = 14 #stacked vertically
    num_loops = 8
    Rwire = 0.0028
    plotZfield(x, y, R, I, d, num_layers, num_loops, Rwire, True)