# nanoowl_utils/setup.py

from setuptools import setup, find_packages

setup(
    name="nanoowl_utils",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "Pillow",
        "numpy",
        "opencv-python",  # Assuming `cv2` is used in `draw_tree_output`
        "nanoowl"
    ],
    entry_points={
        "console_scripts": [
            "process_image=nanoowl_utils.image_processor:main",
        ],
    },
)
