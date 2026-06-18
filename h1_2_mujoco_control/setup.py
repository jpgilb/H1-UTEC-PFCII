from setuptools import find_packages, setup

package_name = 'h1_2_mujoco_control'

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
    maintainer='jpgb',
    maintainer_email='jose.gil@utec.edu.pe',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		'mujoco_follow_joint_trajectory_bridge = h1_2_mujoco_control.mujoco_follow_joint_trajectory_bridge:main',
        ],
    },
)
