from setuptools import setup, find_packages

setup(
    name='haua',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 
        'pandas',
    ],
    author='wuhaohua',
    author_email='wubo.haohua@gmail.com',
    description='...',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_library',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
