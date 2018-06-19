from setuptools import setup, find_packages

setup(
	name='bbknn',
	version='1.0.0',
	description='Batch balanced KNN',
	url='https://github.com/Teichlab/bbknn',
	packages=find_packages(find_packages(exclude=['docs', 'figures'])),
	install_requires=['Cython','numpy','scanpy'],
	author='Krzysztof Polanski, Jongeun Park',
	author_email='kp9@sanger.ac.uk',
	license='MIT'
)