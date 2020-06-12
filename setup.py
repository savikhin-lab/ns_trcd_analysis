from setuptools import setup, find_packages

setup(
    name='ns_trcd_analysis',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'numpy',
        'scipy',
        'matplotlib',
        'h5py',
    ],
    entry_points='''
        [console_scripts]
        ns_trcd_analysis=ns_trcd_analysis.main:cli
    ''',
)
