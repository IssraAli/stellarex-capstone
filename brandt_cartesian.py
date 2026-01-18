import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from constants import *
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors

B_a = 0.003
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
            jdA_x_rprime = np.cross(j_on_quadpoints, rprime) * self.dx * self.dy
            denom = 1/(np.linalg.norm(rprime, axis=1)**3)
            integrand = np.einsum('ij,i->ij',jdA_x_rprime, denom)
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

    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(2, 3, wspace=0.6, width_ratios=[2,1,1], height_ratios=[2,1])

    #############################################################
    # Current plot

    ax1 = fig.add_subplot(gs[0,0])
    #ax1.quiver(x, y, j()[:, 0], j()[:, 1])
    ax1.set_aspect('equal')
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    jnorm = np.linalg.norm(j(), axis=1).reshape(x.shape)
    field = ax1.pcolor(1000*x, 1000*y, jnorm, cmap = 'plasma', norm=colors.LogNorm(vmin=1e1, vmax=jnorm.max()))
    cbar1 = fig.colorbar(field, ax=ax1, extend='max')
    # ax1.streamplot(1000*x, 1000*y, j()[:, 0].reshape(x.shape), j()[:, 1].reshape(x.shape), color='white')
    cbar1.ax.set_ylabel('Current density [A/m$^2$]', rotation=270, labelpad=15)
    ax1.set_aspect('equal')
    ax1.set_title('Current density magnitude on disk\nfrom Brandt, 1998')

    #############################################################
    # B-field plot
    b_norm = 10000*np.linalg.norm(b_field, axis=1).reshape(x_eval.shape) #now in Gauss
    levels = np.linspace(B_a, 10*B_a, 10)

    ax2 = fig.add_subplot(gs[0,1])
    field = ax2.pcolor(1000*x_eval, 1000*z_eval, b_norm, cmap = 'plasma', norm=colors.LogNorm(vmin=b_norm.min(), vmax=b_norm.max()))
    cbar2 = fig.colorbar(field, ax=ax2, extend='max')
    cbar2.ax.set_ylabel('Magnetic flux density [G]', rotation=270, labelpad=15)
    ax2.streamplot(1000*x_eval, 1000*z_eval, b_field[:, 0].reshape(x_eval.shape), b_field[:, 2].reshape(x_eval.shape), color='white')
    ax2.set_aspect('equal')
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('z [mm]')
    rect1 = patches.Rectangle((-1000*A, -1000*D/2), 2*1000*A, 1000*D, fill=True, alpha=0.5, color='black')
    ax2.add_patch(rect1)
    ax2.set_title('$\\boldsymbol{B}$')

    #############################################################
    # Grad B-field plot

    ax3 = fig.add_subplot(gs[0,2])
    normgrad_bnorm = np.sqrt(((b_norm[:-1,1:] - b_norm[:-1,:-1])/dx_eval)**2 + ((b_norm[1:,:-1] - b_norm[:-1,:-1])/dz_eval)**2)/1000
    grad = ax3.pcolor(1000*x_eval[:-1, :-1], 1000*z_eval[:-1, :-1], normgrad_bnorm, norm=colors.LogNorm(vmin=normgrad_bnorm.min(), vmax=normgrad_bnorm.max()))
    cbar3 = fig.colorbar(grad, ax=ax3, extend='max')
    cbar3.ax.set_ylabel('$|\\nabla B|$ [G/mm]', rotation=270, labelpad=15)
    ax3.set_aspect('equal')
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('z [mm]')
    rect2 = patches.Rectangle((-1000*A, -1000*D/2), 2*1000*A, 1000*D, fill=True, alpha=0.5, color='black')
    ax3.add_patch(rect2)
    ax3.set_title('$|\\nabla B|$')

    #############################################################
    # Grad B-field plot

    ax4 = fig.add_subplot(gs[1,0])
    print(f'min x: {np.min(abs(x_eval))}')
    z_cropped = z_eval[:-1, :-1]
    ax4.plot(1000*z_eval[x_eval==np.min(abs(x_eval))], b_norm[x_eval==np.min(abs(x_eval))], label='Biot-Savart from Brandt')
    # zs = np.linspace(0, z_eval_max, 400)
    # dipole_field = (4/(3*np.pi))*(A**3)*((10000*B_a)/(zs**3))
    # ax4.semilogy(1000*zs, dipole_field)
    ax4.set_xlim(0,1000*z_eval_max)
    #ax4.set_ylim(0, np.max(b_norm))
    ax4.set_title('$B$ on axis [G]')
    ax4.set_xlabel('z [mm]')
    ax4.set_ylabel('$B$ on axis [G]')

    ax5 = fig.add_subplot(gs[1,1:])
    print(f'min x: {np.min(abs(x_eval))}')
    z_cropped = z_eval[:-1, :-1]
    ax5.plot(1000*z_cropped[x_eval[:-1, :-1]==np.min(abs(x_eval))], normgrad_bnorm[x_eval[:-1, :-1]==np.min(abs(x_eval))])
    ax5.set_xlim(0,1000*z_eval_max)
    ax5.set_title('$|\\nabla B|$ on axis [G/mm]')
    ax5.set_xlabel('z [mm]')
    ax5.set_ylabel('$|\\nabla B|$ on axis [G/mm]')
    
    plt.savefig('brandt.png', dpi=600)
    #plt.show()



