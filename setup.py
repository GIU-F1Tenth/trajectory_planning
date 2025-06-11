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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Karim',
    maintainer='Fam Shihata',
    maintainer_email='fam@awadlouis.com',
    description='GIU F1Tenth Car\'s trajectory planning package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node_v1 = pure_pursuit_node.pure_pursuit_v1:main',
            'pure_pursuit_node_v2 = pure_pursuit_node.pure_pursuit_v2:main',
            'a_star_exe = path_planner_node.a_star_planner:main',
            'dijkestra_exe = path_planner_node.dijkstra_planner:main',
            'csv_pub_exe = csv_racingline_publisher.csv_path_pub:main',
            'astar_lookahead_pub_exe = astar_lookahead_publisher.astar_lookahead_pub:main'
        ],
    },
)