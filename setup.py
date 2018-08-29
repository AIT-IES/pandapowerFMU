from distutils.core import setup

setup(
	name='pandapowerFMU',
	version='0.1',
	packages=['pandapowerFMU'],
	install_requires=[
		"fmipp",
		"pandas",
		"pandapower",
		"numpy"
	],
	url='https://github.com/AIT-IES/pandapowerFMU',
	license='MIT license',
	author='Benedikt Pesendorfer',
	author_email='benedikt.pesendorfer@ait.ac.at',
	description='generate Functional Mock-up Units (FMUs) for time-domain simulation using pandapower',
	long_description=open('README.md').read()
)
