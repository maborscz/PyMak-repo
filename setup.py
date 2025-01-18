from setuptools import setup
  
setup( 
    name='PyMak', 
    version='0.1', 
    description='AtomCraft Python library', 
    author='Marcus Borscz', 
    author_email='m.borscz@student.unsw.edu.au', 
    packages=['PyMak'],
    package_dir = {'': 'src'},
    include_package_data=True,
    package_data={'': ['data/*']},
)

"""
python==3.11.4
numpy==1.26.4
scipy==1.14.0
matplotlib==3.9.2
plasmapy==2024.10.0
magpylib==5.1.1
PyYAML==6.0.1
"""