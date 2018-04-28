# GameAI-SuperMarioSuperAI
Spring 2018 - Repository for Team Tensor-Dough

Virtual Box Image
=================
	I have set up a virtual machine image with all the 
	dependencies and files already installed to make it 
	easier if you do not want set to set up the 
	environment on your machine.

	A download for the image can be found here:
		link

	Steps to execute:
	- Import the image into virtual box
	- Boot it up and log in. The user password is "1234"
	- cd Desktop/Tensor-dough/gym-rle
	- To train the model
	  - python mario_train.py
	- To run an already trained model
	  - python mario_run.py


Manual Environment Setup
========================

	Follow the steps in the linked PDF:
	- https://drive.google.com/file/d/1N39pOupAee0x1MUa5E57lyTwSdjqUbrA/view?usp=sharing


Dependencies
============
	- Linux Ubuntu 16.04 (may work with other version but untested)
	- Python 2.7
	  - https://www.python.org/download/releases/2.7/
	- Tensorflow 1.7 (works with or without GPU support)
	  - https://www.tensorflow.org/install/
	- Retro Learning Environment (RLE)
	  - https://github.com/nadavbh12/Retro-Learning-Environment
	- libsdl1.2-dev | libsdl-gfx1.2-dev | libsdl-image1.2-dev | cmake
	  - sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
	- Gym Retro Learning Environment (Gym-RLE)
	  - https://github.com/nadavbh12/gym-rle
	- pip 8.1.1
	  - sudo apt-get install python-pip
	- numpy 1.11.0
	  - sudo apt-get install python-numpy
	- Super Mario World ROM
	  - https://drive.google.com/file/d/154rlFZGZHU2fjPF96K6gzPp9Uszk11k8/view?usp=sharing
	- Keras 2.1.6
	  - sudo pip install keras
	- Keras-rl 0.3.1
	  - sudo pip install keras-rl==0.3.1
	- Pillow
	  - sudo pip install pillow
	- h5py
	  - sudo pip install h5py




