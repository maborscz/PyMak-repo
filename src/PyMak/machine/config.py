from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import iv
import matplotlib.pyplot as plt
from matplotlib import patches
import yaml
from numpy.typing import ArrayLike


class MachineConfig():
    """Configuration of machine geometry (coils and vaccuum vessel)

    Parameters
    ----------
    name : str
        Name of machine configuration
    dirpath : str, optional
        Path to directory for saving machine information, by default None

    Attributes
    ----------
    name
    dirpath
    pfdata : dict
        Data for each PF coil (r, z, width, height, number of turns, cable diameter)
    tfdata : dict
        Data for the D-shaped TF coils (rmin, rmax, number of turns, number of coils, cable diameter)
    tf_dee : ArrayLike
        (n, 2) array containing (x, z) coordinates of the TF Dee guideline
    tf_windings : ArrayLike
        (m, n, 3) array containing (x, y, z) coordinates of each TF winding
    pf_windings : dict
        Dictionary indexed by PF coil name, with each item a (n, 2) array containing (r, z) coordinates of each winding
    """

    def __init__(self,
        name: str,
        dirpath: str = None,
    ):
        self.name = name

        if dirpath is None:
            self.dirpath = Path(__file__).parent.parent / 'data' / 'machine' / f'{name}'
        else:
            self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

        self.pfdata = {}
        self.pf_windings = {}


    def create_tf_coil(self,
        rmin: float,
        rmax: float,
        nturns: int,
        ncoils: int = 8,
        D: float = 11.9e-3,
        N1: int = 50,
        N2: int = 5
    ):
        """Define the TF coil parameters and create the windings, assuming triangular packing

        Parameters
        ----------
        rmin : float
            Radius of TF inner leg
        rmax : float
            Radius of TF outer leg
        nturns : int
            Number of turns per TF coil
        ncoils : int, optional
            Number of TF coils, by default 8
        D : float, optional
            TF cable diameter, by default 11.9e-3
        N1 : int, optional
            Number of points to represent curved section, by default 50
        N2 : int, optional
            Number of points to represent straight section, by default 5
        """
        # Princeton Dee parameters
        r0 = np.sqrt(rmin * rmax)
        k = 1/2 * np.log(rmax/rmin)

        # Solve Princeton Dee differential equation in polar form
        func = lambda t, y: [y[1], (y[0]**2 + 2*y[1]**2 - (y[0]**2 + y[1]**2)**(3/2)/(k*(y[0]*np.cos(t) + rmin)))/y[0]]
        t = np.linspace(0, np.pi/2, N1)
        rho = solve_ivp(func, [t[0], t[-1]], [rmax - rmin, 0], t_eval=t, method='RK45').y[0]

        # Create (r, z) cooridnates for full Dee shape
        r, z = rho * np.cos(t) + rmin, rho * np.sin(t)
        r = np.concatenate([r, np.full(N2, rmin), r[:0:-1]])
        z = np.concatenate([z, np.linspace(1, -1, N2) * np.pi * k * iv(1, k) * r0, -z[:0:-1]])

        # Calculate angle of dz/dr gradient
        theta = np.arctan2(-np.gradient(z), -np.gradient(r))

        # Calculate size and number of layers for triangular packing
        li = np.ceil((np.sqrt(8*nturns + 1) - 1)/2)
        lf = np.floor((np.sqrt(4*(li*(li + 1) - 2*nturns) + 1) - 1)/2)
        extra = li*(li + 1)/2 - lf*(lf + 1)/2 - nturns

        # Calculate (r, z) cooridnates of each winding, assuming triangular packing
        windings = []
        x0 = np.sqrt((rmin - D/2)**2 - ((lf - 1) * D/2)**2)
        for i in range(int(li), int(lf), -1):
            y0 = sorted(np.linspace(-D, D, i) * (i - 1)/2, key=np.abs)
            if i == lf + 1 and extra > 0:
                y0 = y0[:-int(extra)]

            for _y0 in y0:
                # Windings are assumed to be expansions around guideline
                winding = np.empty((len(r), 3))
                winding[:, 0] = r - (rmin - x0) * np.sin(theta)
                winding[:, 1] = _y0
                winding[:, 2] = z + (rmin - x0) * np.cos(theta)
                windings.append(winding)

            x0 -= D * np.sqrt(3)/2

        self.tfdata = {
            'ncoils': int(ncoils),
            'nturns': int(nturns),
            'rmin': float(rmin),
            'rmax': float(rmax),
            'D': float(D),
        }
        self.tf_windings = np.array(windings)
        self.tf_dee = np.stack([r, z], axis=-1)


    def create_cs_coil(self,
        rmin: float,
        nturns: int,
        nlayers: int,
        D: float = 4.85e-3,
    ):
        """Define the CS parameters and create the windings, assuming triangular packing

        Parameters
        ----------
        rmin : float
            Inner radius of solenoid
        nturns : int
            Number of turns
        nlayers : int
            Number of winding layers
        D : float, optional
            CS cable diameter, by default 4.85e-3
        """
        width = ((nlayers - 1) * np.sqrt(3)/2 + 1) * D
        rc = rmin + width/2

        self.create_pf_coil('CS', rc, 0, nturns, nlayers, D)

    
    def create_pf_coil(self,
        name: str,
        rc: float ,
        zc: float,
        nturns: int,
        nlayers: int,
        D: float = 4.85e-3,
    ):
        """Define the parameters for a PF coil and create the windings, assuming triangular packing

        Parameters
        ----------
        name : str
            Name of PF coil
        rc : float
            Coil centre radius
        zc : float
            Coil centre height
        nturns : int
            Number of turns
        nlayers : int
            Number of winding layers
        D : float, optional
            PF cable diameter, by default 4.85e-3
        """
        n = np.ceil((nturns + np.floor(nlayers/2))/nlayers)
        extra = n * nlayers - np.floor(nlayers/2) - nturns

        width = ((nlayers - 1) * np.sqrt(3)/2 + 1) * D
        height = n * D
        r0 = -(width - D)/2

        # Calculate winding positions, assuming triangular packing
        r, z = [], []
        for i in range(nlayers):
            if i % 2 == 0: m = n
            else: m = n - 1
            
            z0 = sorted(np.linspace(-D, D, int(m)) * (m - 1)/2, key=np.abs)
            if i == nlayers - 1 and extra > 0:
                z0 = z0[:-int(extra)]
            
            r += [r0] * len(z0)
            z += z0

            r0 += D * np.sqrt(3)/2

        windings = np.stack([r, z], axis=-1) + np.array([rc, zc])

        self.pfdata[name] = {
            'nturns': int(nturns),
            'rc': float(rc),
            'zc': float(zc),
            'w': float(width),
            'h': float(height),
            'D': float(D),
        }
        self.pf_windings[name] = windings

    
    def create_vessel(self,
        r0: float = 0.316,
        OD: float = 0.162,
        t: float = 0.01,
    ):
        """Create vacuum vessel parameters, asumming it has a circular cross-section

        Parameters
        ----------
        r0 : float, optional
            Major radius, by default 0.316
        OD : float, optional
            Outer diamater, by default 0.162
        t : float, optional
            Wall thickness, by default 0.01
        """
        self.vvdata = {
            'r0': float(r0),
            'OD': float(OD),
            't': float(t),
        }


    def plot_machine(self):
        """Plot coils and vacuum vessel and save to `dirpath`
        """
        fig, ax = plt.subplots(1, 1, figsize=(12,10))
        ax.set_aspect('equal')
        
        if hasattr(self, 'tf_windings'):
            for tf in self.tf_windings:
                ax.plot(*tf.T[[0, 2]], color='purple')
            ax.plot(*self.tf_guideline.T[[0, 2]], ls='--', color='purple', label='TF Dee')

        if hasattr(self, 'vvdata'):
            vessel = patches.Annulus((self.vvdata['r0'], 0), self.vvdata['OD'], self.vvdata['t'], fill=True, color='silver', label='VV')
            ax.add_patch(vessel)

        for i, (name, pf) in enumerate(self.pf_windings.items()):
            for (r, z) in pf:
                circ = patches.Circle((r, z), radius=self.pfdata[name]['D']/2, color='purple')
                ax.add_patch(circ)

            xy = (self.pfdata[name]['rc'] - self.pfdata[name]['w']/2, self.pfdata[name]['zc'] - self.pfdata[name]['h']/2)
            rect = patches.Rectangle(xy, self.pfdata[name]['w'], self.pfdata[name]['h'], linewidth=2, edgecolor=f'C{i}', facecolor='none', label=name)
            ax.add_patch(rect)

        ax.set_xlabel(r'$r$ (m)')
        ax.set_ylabel(r'$z$ (m)')
        ax.set_xlim(left=0)
        plt.legend(loc='upper right')
        plt.savefig(self.dirpath / 'machine.png', bbox_inches='tight', dpi=1200)
        

    def save(self):
        """Save machine configuration to a .yaml file, windings to a .npz file and plot the machine cross-section
        """
        data = {
            'VV': self.vvdata,
            'TF': self.tfdata,
            'PF': self.pfdata
        }
        with open(self.dirpath / 'cfg.yaml', 'w') as f:
            yaml.dump(data, f)
        
        np.savez(self.dirpath / 'windings.npz', TF=self.tf_windings, PF=self.pf_windings)

        self.plot_machine()



if __name__ == '__main__':
    cfg = MachineConfig('test')
    cfg.create_tf_coil(0.125, 0.55, 10)
    cfg.create_cs_coil(0.055, 185, 3)
    cfg.create_pf_coil('PF1U', 0.18, 0.20, 15, 6)
    cfg.create_pf_coil('PF1L', 0.18, -0.20, 15, 6)
    cfg.create_pf_coil('PF2U', 0.44, 0.20, 5, 3)
    cfg.create_pf_coil('PF2L', 0.44, -0.20, 5, 3)
    cfg.create_pf_coil('PF3U', 0.5, 0.13, 5, 3)
    cfg.create_pf_coil('PF3L', 0.5, -0.13, 5, 3)
    cfg.create_vessel()

    cfg.save()