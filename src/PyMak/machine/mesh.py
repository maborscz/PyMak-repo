import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain, save_gs_mesh


# TODO: Fix documentation

class TokaMakerMesh():
    def __init__(self,
        name: str = 'SOUTH',
        plasma_dx: float = 5e-3,
        coil_dx: float = 5e-3,
        vv_dx: float = 4e-3,
        vac_dx: float = 1e-2,
        noncontinuous: bool = False,
        num_circle_pts: int = 100,
    ) -> None:
        
        self.dirpath = Path(__file__).parent.parent / 'data' / 'machine' / f'{name}'

        with open(self.dirpath / 'cfg.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

        self.mesh = gs_Domain(rextent=cfg['rmax'], zextents=[-cfg['zmax'], cfg['zmax']])
        self.mesh.define_region('air', vac_dx, 'boundary')
        self.mesh.define_region('plasma', plasma_dx, 'plasma')
        self.mesh.define_region('vv', vv_dx, 'conductor', eta=6.9E-7, noncontinuous=noncontinuous)

        R0, a, t = cfg['limiter']['major_radius'], cfg['limiter']['minor_radius'], cfg['limiter']['thickness']
        theta = np.linspace(0, 2*np.pi, num_circle_pts)
        inner_vv = np.stack([R0 + a * np.cos(theta), a * np.sin(theta)], axis=1).tolist()
        outer_vv = np.stack([R0 + (a+t) * np.cos(theta), (a+t) * np.sin(theta)], axis=1).tolist()
        self.mesh.add_annulus(inner_vv, 'plasma', outer_vv, 'vv', parent_name='air')

        for name, coil in cfg['PF'].items():
            if 'coilset' in coil.keys():
                self.mesh.define_region(name, coil_dx, 'coil', coil_set=coil['coilset'], nTurns=coil['nturns'])
            else:
                self.mesh.define_region(name, coil_dx, 'coil', nTurns=coil['nturns'])
            self.mesh.add_rectangle(coil['r'], coil['z'], coil['width'], coil['height'], name, parent_name='air')


    def save(self, savename):
        savepath = self.dirpath / 'meshes' / savename
        savepath.mkdir(parents=True, exist_ok=True)

        mesh_pts, mesh_lc, mesh_reg = self.mesh.build_mesh()
        coil_dict = self.mesh.get_coils()
        cond_dict = self.mesh.get_conductors()

        save_gs_mesh(mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict, savepath / 'mesh.h5')

        fig, ax = plt.subplots(2, 2, figsize=(8,8), constrained_layout=True)
        self.mesh.plot_mesh(fig, ax)
        plt.savefig(savepath / 'meshplot.png')


if __name__ == '__main__':
    mesh = TokaMakerMesh()
    mesh.save('MESH001')