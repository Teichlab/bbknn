from setuptools import setup, find_packages

setup(
	name='bbknn',
	version='1.3.11',
	description='Batch balanced KNN',
	url='https://github.com/Teichlab/bbknn',
	packages=find_packages(exclude=['docs', 'figures', 'examples']),
	install_requires=['Cython','numpy','scipy','annoy','umap-learn','scikit-learn'],
	extras_require=dict(
		faiss=['faiss']
	),
	author='Krzysztof Polanski, Jongeun Park',
	author_email='kp9@sanger.ac.uk',
	license='MIT'
)