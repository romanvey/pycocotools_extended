import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='pycocotools_extended',
    version='1.0.0',
    author='Roman Vey & Vasyl Borsuk',
    author_email='roman.vey@gmail.com, vas.borsuk@gmail.com',
    description='Extended pycocoutils',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/romanvey/pycocotools_extended',
    packages=setuptools.find_packages(exclude=['examples']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=required
)
