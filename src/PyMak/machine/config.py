import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.special import iv
import yaml

from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain, save_gs_mesh, load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun


# TODO: Fix documentation

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

        self.PF_colour = 0
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12,10))
        self.ax.set_aspect('equal')
        self.ax.set_xlabel(r'$r$ (m)')
        self.ax.set_ylabel(r'$z$ (m)')

        self.rmax, self.zmax = 0, 0


    def create_tf_coil(self,
        r1: float = 0.12,
        r2: float = 0.55,
        r1_to_inner: float = 0.01626,
        r1_to_outer: float = 0.01626,
        nturns: int = 10,
        ncoils: int = 8,
        N1: int = 50,
        N2: int = 5,
        sldcrv: bool = True,
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
        r0 = np.sqrt(r1 * r2)
        k = 1/2 * np.log(r2/r1)

        # Solve Princeton Dee differential equation in polar form
        func = lambda t, y: [y[1], (y[0]**2 + 2*y[1]**2 - (y[0]**2 + y[1]**2)**(3/2)/(k*(y[0]*np.cos(t) + r1)))/y[0]]
        t = np.linspace(0, np.pi/2, N1)
        rho = solve_ivp(func, [t[0], t[-1]], [r2 - r1, 0], t_eval=t, method='RK45').y[0]

        # Create (r, z) coordinates for full Dee shape
        r, z = rho * np.cos(t) + r1, rho * np.sin(t)
        r = np.concatenate([r, np.full(N2, r1), r[:0:-1]])
        z = np.concatenate([z, np.linspace(1, -1, N2) * np.pi * k * iv(1, k) * r0, -z[:0:-1]])
        self.tf_dee = np.stack([r, z], axis=-1)

        # Calculate angle of dz/dr gradient
        theta = np.arctan2(-np.gradient(r), np.gradient(z))

        self.tf_inner = self.tf_dee.copy()
        self.tf_inner[:, 0] -= r1_to_inner * np.cos(theta)
        self.tf_inner[:, 1] -= r1_to_inner * np.sin(theta)

        self.tf_outer = self.tf_dee.copy()
        self.tf_outer[:, 0] += r1_to_outer * np.cos(theta)
        self.tf_outer[:, 1] += r1_to_outer * np.sin(theta)

        self.rmax = max(self.rmax, np.max(self.tf_outer[:, 0]))
        self.zmax = max(self.zmax, np.max(np.abs(self.tf_outer[:, 1])))

        self.ax.add_patch(patches.Polygon(self.tf_dee, fill=False, ls='-', color='purple', label='TF Centroid'))
        self.ax.add_patch(patches.Polygon(self.tf_outer, fill=False, ls='--', color='purple', label='TF Boundary'))
        self.ax.add_patch(patches.Polygon(self.tf_inner, fill=False, ls='--', color='purple'))
            
        self.tfdata = {
            'ncoils': int(ncoils),
            'nturns': int(nturns),
            'r1': float(r1),
            'r2': float(r2),
        }

        if sldcrv == True:
            self._save_tf_as_sldcrv('dee')
            self._save_tf_as_sldcrv('inner')
            self._save_tf_as_sldcrv('outer')
            

    def _save_tf_as_sldcrv(self,
        curve: str,
    ) -> None:
        if curve == 'dee':
            r, z = self.tf_dee.T
        elif curve == 'inner':
            r, z = self.tf_inner.T
        elif curve == 'outer':
            r, z = self.tf_outer.T

        X = np.char.add(r.astype('str'), 'm')
        Y = np.char.add(z.astype('str'), 'm')
        Z = np.char.add(np.zeros_like(r).astype('str'), 'm')

        df = pd.DataFrame([X, Y, Z]).T
        df.to_csv(self.dirpath / f'tf_{curve}.sldcrv', index=None, header=None, sep=' ')


    def create_cs_coil(self,
        r: float = 0.0846,
        width: float = 0.0091,
        height: float = 0.3929,
        nturns: int = 161,
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
        self.create_pf_coil('CS', r, 0, width, height, nturns)

    
    def create_pf_coil(self,
        name: str,
        r: float ,
        z: float,
        width: float,
        height: float,
        nturns: int = 1,
        coilset: str = None,
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
        xy = (r - width/2, z - height/2)
        self.ax.add_patch(patches.Rectangle(xy, width, height, color=f'C{self.PF_colour}', label=name))
        self.PF_colour += 1
        self.rmax = max(self.rmax, r + width/2)
        self.zmax = max(self.zmax, np.abs(z) + height/2)

        self.pfdata[name] = {
            'nturns': nturns,
            'r': r,
            'z': z,
            'width': width,
            'height': height,
        }

        if coilset is not None:
            self.pfdata[name]['coilset'] = coilset

    
    def create_vessel(self):
        """Create vacuum vessel parameters, assuming it has a circular cross-section.
        """
        # All data taken from Master CAD
        major_radius, minor_radius, thickness = 0.3156, 0.1574, 0.0046
        self.limiter_data = {
            'major_radius': major_radius,
            'minor_radius': minor_radius,
            'thickness': thickness,
        }
        
        self.rmax = max(self.rmax, 0.4776)
        self.zmax = max(self.zmax, 0.162)

        self.ax.add_patch(patches.Annulus((major_radius, 0), minor_radius + thickness, thickness, fc='silver', label='VV'))

        self.ax.add_patch(patches.Circle((major_radius, 0), 0.2080, ec='black', fill=False, ls=':', label='Flange Boundary'))

        self.ax.plot((0.4345, 0.5076), (0.11, 0.11), color='black', ls=':')
        self.ax.plot((0.4345, 0.5076), (-0.11, -0.11), color='black', ls=':')
        self.ax.add_patch(patches.Rectangle((0.5076, -0.143), 0.077, 0.286, ec='black', fill=False, ls=':'))

        self.ax.plot((0.3965, 0.3965), (0.1404, 0.1883), color='black', ls=':')
        self.ax.plot((0.2436, 0.2436), (0.1451, 0.1883), color='black', ls=':')
        self.ax.add_patch(patches.Rectangle((0.2071, 0.1883), 0.226, 0.077, ec='black', fill=False, ls=':'))

        self.ax.plot((0.3965, 0.3965), (-0.1404, -0.1883), color='black', ls=':')
        self.ax.plot((0.2436, 0.2436), (-0.1451, -0.1883), color='black', ls=':')
        self.ax.add_patch(patches.Rectangle((0.2071, -0.1883), 0.226, -0.077, ec='black', fill=False, ls=':'))


    def plot_machine(self):
        """Plot coils and vacuum vessel and save to `dirpath`
        """
        self.ax.set_xlim(0, self.rmax*1.2)
        self.ax.set_ylim(-self.zmax*1.2, self.zmax*1.2)
        self.ax.legend(loc='upper right')

        plt.savefig(self.dirpath / 'machine.png', bbox_inches='tight', dpi=1200)
        

    def save_cfg(self):
        """Save machine configuration to a .yaml file and plot the machine cross-section
        """
        self.create_vessel()
        self.plot_machine()

        data = {
            'limiter': self.limiter_data,
            'TF': self.tfdata,
            'PF': self.pfdata,
            'rmax': round(float(self.rmax), 3),
            'zmax': round(float(self.zmax), 3),
        }
        with open(self.dirpath / 'cfg.yaml', 'w') as f:
            yaml.dump(data, f)
        

    def save_mesh(self,
        plasma_dx: float = 5e-3,
        coil_dx: float = 1e-2,
        vv_dx: float = 1e-2,
        vac_dx: float = 1e-2,
        noncontinuous: bool = False,
    ):
        self.mesh = gs_Domain(rextent=self.rmax, zextents=[-self.zmax, self.zmax])
        self.mesh.define_region('air', vac_dx, 'boundary')
        self.mesh.define_region('plasma', plasma_dx, 'plasma')
        self.mesh.define_region('vv', vv_dx, 'conductor', eta=6.9E-7, noncontinuous=noncontinuous)



if __name__ == '__main__':
    cfg = MachineConfig('SOUTH')
    cfg.create_tf_coil()
    cfg.create_cs_coil()
    cfg.create_pf_coil('PF1U', 0.18, 0.18, 0.0267, 0.0133, nturns=15, coilset='PF1')
    cfg.create_pf_coil('PF2U', 0.44, 0.18, 0.0121, 0.0133, nturns=6, coilset='PF2')
    cfg.create_pf_coil('PF3U', 0.49, 0.13, 0.0121, 0.0133, nturns=6, coilset='PF3')
    cfg.create_pf_coil('PF1L', 0.18, -0.18, 0.0267, 0.0133, nturns=15, coilset='PF1')
    cfg.create_pf_coil('PF2L', 0.44, -0.18, 0.0121, 0.0133, nturns=6, coilset='PF2')
    cfg.create_pf_coil('PF3L', 0.49, -0.13, 0.0121, 0.0133, nturns=6, coilset='PF3')    

    cfg.save_cfg()