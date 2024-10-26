from setuptools import setup, find_packages


def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


reqs = parse_requirements('requirements.txt')

setup(name='onnx_opcounter',
      version='0.0.4',
      description='ONNX flops / params counter',
      author='Grigory Malivenko',
      author_email='',
      packages=find_packages(),
      install_requires=reqs,
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'onnx_opcounter = onnx_opcounter.cli:main',
          ],
      },
)
