import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from constants import *
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors

B_a = 0.0005
B_c = 1.6
A = 0.02
D = 0.006
r_p = A/np.cosh(B_a/B_c)
c = np.tanh(B_a/B_c)
J_c = B_c*2/(D*mu0)
print(f'Bean critical current: {J_c} [A/m^2]')

SMOKE_TEST = False

class CurrentDist:
    def __init__(self, current: callable, quadpoints:np.ndarray, dx, dy):
        self.current = current
        self.quadpoints = quadpoints
        self.dx = dx
        self.dy = dy
        self._cache = {'j_on_quadpoints':self.current(quadpoints)}
    def __call__(
            self,
    ):
        return self._cache['j_on_quadpoints']
    def biot_savart(
            self,
            eval_points,
        ):
        b_field = np.zeros_like(eval_points)
        for i, eval_point in enumerate(eval_points):
            j_on_quadpoints = self._cache['j_on_quadpoints']
            rprime = eval_point - self.quadpoints
            # assuming uniform spacing in x and y for quadpoints
            jdA_x_rprime = np.cross(j_on_quadpoints, rprime) * self.dx * self.dy
            # print(f'np.all(jdA_x_rprime==0): {np.all(jdA_x_rprime==0)}')
            # print(f'jdA_x_rprime: {jdA_x_rprime}')
            # print(f'jdA_x_rprime.shape: {jdA_x_rprime.shape}')
            denom = 1/(np.linalg.norm(rprime, axis=1)**3)
            # print(f'denom.shape: {denom.shape}')
            integrand = np.einsum('ij,i->ij',jdA_x_rprime, denom)
            # print(f'integrand.shape: {integrand.shape}')
            b_field[i,:] = (mu0/(4*np.pi))*np.sum(integrand, axis=0)
        b_field[:, 2] -= B_a
        return b_field


def brandt_current(eval_point):
    assert np.all(eval_point[:,2]==0)
    r = np.linalg.norm(eval_point, axis=1)
    j = np.zeros_like(r)
    j[r<=r_p] = J_c * (2/np.pi) * np.arctan((c*r[r<=r_p])/np.sqrt(r_p**2 - r[r<=r_p]**2))
    j[r>r_p] = J_c
    j[r>A] = 0
    e_phi = np.array([-quadpoints[:,1]/r, quadpoints[:,0]/r, 0*r]).T
    return np.einsum('i,ij->ij', j, e_phi)
    

if __name__ == "__main__":
    xmin = -A
    xmax = A
    ymin = -A
    ymax = A
    nx = 80 if SMOKE_TEST else 200
    ny = 80 if SMOKE_TEST else 200
    nz = 80 if SMOKE_TEST else 200

    x1d, y1d = np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax,ny)
    x, y = np.meshgrid(x1d, y1d)
    dx = x1d[1] - x1d[0]
    dy = y1d[1] - y1d[0]
    z = np.zeros_like(x)
    quadpoints = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    j = CurrentDist(brandt_current, quadpoints, dx, dy)

    x_eval_min = -(1.5)*A
    x_eval_max = (1.5)*A
    z_eval_min = -(6)*A
    z_eval_max = (6)*A
    nx_eval = 80 if SMOKE_TEST else 200
    nz_eval = 80 if SMOKE_TEST else 200
    x_eval_1d, z_eval_1d = np.linspace(x_eval_min,x_eval_max,nx_eval), np.linspace(z_eval_min,z_eval_max,nz_eval)
    dx_eval = x_eval_1d[1] - x_eval_1d[0]
    dz_eval = z_eval_1d[1] - z_eval_1d[0]
    x_eval, z_eval = np.meshgrid(x_eval_1d, z_eval_1d)
    y_eval = np.zeros_like(x_eval)
    eval_points = np.vstack([x_eval.flatten(), y_eval.flatten(), z_eval.flatten()]).T

    b_field = j.biot_savart(eval_points)

    fig = plt.figure(figsize=(18,6))
    gs = GridSpec(1, 3, wspace=0.1)

    #############################################################
    # Current plot

    ax1 = fig.add_subplot(gs[0,0])
    ax1.quiver(x, y, j()[:, 0], j()[:, 1])
    ax1.set_aspect('equal')
    ax1.set_xlabel('x[m]')
    ax1.set_ylabel('y[m]')

    b_norm = np.linalg.norm(b_field, axis=1).reshape(x_eval.shape)

    #############################################################
    # B-field plot

    levels = np.linspace(B_a, 10*B_a, 10)

    ax2 = fig.add_subplot(gs[0,1])
    ax2.contourf(x_eval, z_eval, b_norm, cmap = 'plasma')
    ax2.streamplot(x_eval, z_eval, b_field[:, 0].reshape(x_eval.shape), b_field[:, 2].reshape(x_eval.shape), color='white')
    ax2.set_aspect('equal')

    rect = patches.Rectangle((-A, -D/2), 2*A, D, fill=True, alpha=0.5, color='white')
    ax2.add_patch(rect)

    #############################################################
    # Grad B-field plot

    ax3 = fig.add_subplot(gs[0,2])
    normgrad_bnorm = np.sqrt(((b_norm[:-1,1:] - b_norm[:-1,:-1])/dx_eval)**2 + ((b_norm[1:,:-1] - b_norm[:-1,:-1])/dz_eval)**2)
    ax3.contourf(x_eval[:-1, :-1], z_eval[:-1, :-1], 20*np.log10(normgrad_bnorm))

    plt.show()



