from setuptools import find_packages, setup

package_name = 'h1_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sebas',
    maintainer_email='sebastian70081@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'object_tracker = h1_vision.object_tracker:main',
            'depth_test = h1_vision.depth_test:main',
            'sphere_tracker = h1_vision.sphere_tracker:main',
            'aruco_tracker = h1_vision.aruco_tracker:main'
        ],
    },
)
