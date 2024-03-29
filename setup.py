import versioneer
from setuptools import setup, find_packages

setup(
    name="ch_pipeline",
    packages=find_packages(),
    package_data={"ch_pipeline": ["data/*"]},
    install_requires=["Click"],
    extras_require={
        "hfb": ["beam-model @ git+ssh://git@github.com/CHIMEFRB/beam-model.git"],
    },
    python_requires=">=3.9",
    entry_points="""
        [console_scripts]
        chp=ch_pipeline.processing.client:cli
    """,
    author="CHIME collaboration",
    author_email="richard@phas.ubc.ca",
    description="CHIME Pipeline",
    url="http://github.com/chime-experiment/ch_pipeline/",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
)
