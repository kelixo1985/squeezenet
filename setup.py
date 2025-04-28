from setuptools import setup, find_packages

setup(
    name='keras-squeezenet',
    version='1.0',
    description='SqueezeNet v1.0 model for Keras and TensorFlow',
    author='Nama Anda',
    author_email='kelixo@gmail.com',
    url='https://github.com/kelixo1985/squeezenet',
    packages=find_packages(),
    install_requires=[
        'tensorflow',  # tensorflow version>=2.8
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Acilovin License',
        'Operating System :: OS Independent',
    ],
)