import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="simulate_glitches",
    version="0.0.1",
    author="Beth Flanagan",
    author_email="bethflanagan20@gmail.com",
    description=("Simulate LISA data with glitches and gaps"),
    packages=find_packages(),
    entry_points={'console_scripts': [
                'make-lisa-glitches=simulate_glitches.make_glitch:main',
                'inject-lisa-glitches=simulate_glitches.inject_glitch:main']},
    long_description=read('README'),

)