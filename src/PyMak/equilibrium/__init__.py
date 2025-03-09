import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh


# TODO: Add documentation and make it actually work!


class TokaMakerEquilibrium():

    def __init__(self, mesh, machine='SOUTH', B0=0.1, Ip=5e3, pax=5e1):
        self.B0 = B0
        self.Ip = Ip
        self.pax = pax

        self.dirpath = Path(__file__).parent.parent / 'data' / 'machine' / f'{machine}' / 'meshes' / f'{mesh}' / 'mesh.h5'

        self.mygs = TokaMaker()

        mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(f'meshes/{self.meshname}.h5')
        self.mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        self.mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
        
        coil_bounds = np.zeros((self.mygs.ncoils+1,2), dtype=np.float64)
        coil_bounds[:,0] = -1e3; coil_bounds[:,1] = 1e3
        self.mygs.set_coil_bounds(coil_bounds)

        ffp_prof = create_power_flux_fun(40, 1.5, 2.0)
        pp_prof = create_power_flux_fun(40, 4.0, 1.0)
        self.mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

        self.mygs.set_targets(Ip=self.Ip, pax=self.pax)
    

    def setup_LCFS(self, R0=0.3, Z0=0, a=0.08, kappa=1.5, delta=0.15, xi=0, num=10, xpoints=None):
        self.mygs.setup(order=2, F0=self.B0 * R0)

        theta = np.linspace(0, 2 * np.pi, num, endpoint=False)
        R = R0 + a * np.cos(theta + delta * np.sin(theta) - xi * np.sin(2 * theta))
        Z = Z0 + kappa * a * np.sin(theta + xi * np.sin(2 * theta))
        self.mygs.set_isoflux(np.stack([R, Z], axis=1))

        if xpoints is not None:
            self.mygs.set_saddles(xpoints)

        self.mygs.init_psi(R0, Z0, a, kappa, delta)


    def solve(self):
        self.mygs.solve()

        fig, ax = plt.subplots(1,1)
        self.mygs.plot_machine(fig, ax, coil_colormap='seismic', coil_symmap=True, coil_scale=1.E-3, coil_clabel=r'$I_C$ [kA]')
        self.mygs.plot_psi(fig, ax, xpoint_color=None, vacuum_nlevels=4)
        self.mygs.plot_constraints(fig, ax, isoflux_color='tab:red', isoflux_marker='.')

        self.mygs.print_info()

        print()
        print("Coil Currents [kA]:")
        coil_currents, _ = self.mygs.get_coil_currents()
        for key in self.mygs.coil_sets:
            i = self.mygs.coil_sets[key]['id']
            print('  {0:10} {1:10.2F}'.format(key+":",coil_currents[i]/1.E3))