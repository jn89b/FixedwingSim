import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fixedwing_sim",
    version="0.0.1",
    author="Justin Nguyen",
    author_email="jnguyenblue2804@gmail.com",
    # author_email="aq15777@bristol.ac.uk",
    # description="A python programme to accomplish various ML tasks with FW aircraft in AirSim",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="https://github.com/AOS55/Fixedwing-Airsim",
    # project_urls={
    #     "Bug Tracker": "https://github.com/AOS55/Fixedwing-Airsim/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
)


