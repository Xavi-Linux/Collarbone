from setuptools import find_namespace_packages, setup

setup(name='tftools',
      version='0.0.1',
      description='A package to ease Tensorflow DNNs visualization',
      long_description='A test to learn to package',
      url='https://github.com/Xavi-Linux/Collarbone/tree/dev/tftools/tftools',
      author='xavi-linux',
      author_email='xavi.ms@xavims.net',
      keywords='tftools',
      packages=find_namespace_packages(include=['tftools.*']),
      python_requires='>=3.6, <4',
      install_requires=['tensorflow>=2.7.0',
                        'matplotlib>=3.4.0',
                        'numpy>=1.19.5',
                        'ipympl'
                        ]
     )