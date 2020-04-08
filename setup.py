# python 

from setuptools import setup


setup(
  name = 'TensorFI',
  packages = ['TensorFI'],
  version = '2.0.0',
  license='MIT',
  description = 'A fault injection tool for TensorFlow-based program',
  author = 'DependableSystemsLab',
  author_email = 'karthikp@ece.ubc.ca',
  url = 'https://github.com/DependableSystemsLab/TensorFI',
  download_url = 'https://github.com/DependableSystemsLab/TensorFI/archive/v2.0.0.tar.gz',
  long_description= "TensorFI is a fault injection framework for injecting both hardware and software faults into applications written using the TensorFlow framework.   \
                      Check the GitHub repo for more details",
  install_requires=[
          'pyyaml',
          'scikit-learn',
          'tensorflow',
          'numpy',
          'enum34',
      ],
  classifiers=[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
#    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
#    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 2.7',
  ],
)

