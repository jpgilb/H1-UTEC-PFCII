from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'h1_2_model'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Oscar E. Ramos',
    maintainer_email='oramos@utec.edu.pe',
    description='Kinematic and dynamic tests for H1-2 Model',
    license='TODO',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'joint_publisher = h1_2_model.joint_publisher:main',
            'fk_test_position = h1_2_model.fk_test_position:main',
            'fk_test_pose = h1_2_model.fk_test_pose:main',
            'jacobian_test = h1_2_model.jacobian_test:main',
            'jacobian_numeric_test = h1_2_model.jacobian_numeric_test:main',
            'kinematic_controller_fixed = h1_2_model.kinematic_controller_fixed:main',
            'kinematic_controller_wb1 = h1_2_model.kinematic_controller_wb1:main',
            'kinematic_controller_wb2 = h1_2_model.kinematic_controller_wb2:main',
        ],

    },
)
