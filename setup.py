from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'trajectory_planning'

setup(
    name=package_name,
    version='0.2.3',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include path files
        (os.path.join('share', package_name, 'path'), glob('path/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Fam Shihata',
    author_email='fam@awadlouis.com',
    maintainer='Fam Shihata, George Halim',
    maintainer_email='fam@awadlouis.com, georgehany064@gmail.com',
    description='A comprehensive ROS2 package for autonomous vehicle trajectory planning and path following',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "csv_path_pub = global_planner.csv_path_publisher:main",
            'dynamic_lookahead_pub_exe = local_planner.dynamic_lookahead:main'
        ],
    },
)
