from distutils.core import setup

setup(
    name='nn_pipeline',  # How you named your package folder (MyLib)
    packages=[
        'nnutils',
        'nnutils.cnn_complexity_analyzer',
        'nnutils.training_pipeline',
        'nnutils.training_pipeline.trainers',
    ],  # Chose the same as "name"
    version='0.7',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='Common utility functions for training and evaluating neural networks as well as analyzing computational complexity',
    # Give a short description about your library
    author='Rex Geng',  # Type in your name
    author_email='hgeng4@illinois.edu',  # Type in your E-Mail
    url='https://github.com/rexxy-sasori/nnutils.git',
    # Provide either the link to your github or to your website
    download_url='https://github.com/rexxy-sasori/nnutils/archive/refs/tags/v0.7.tar.gz',  # I explain this later on
    keywords=['Complexity', 'Profiler', 'CNN'],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        'torch>=1.8.1',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.7',
    ],
)
