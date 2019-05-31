
#Track list: aalborg, dirt, dirt-5, e-track-1, e-track-4, g-track-1, mixed-2, wheel-1, a-speedway, dirt-2, dirt-6, e-track-2, e-track-5, oval, dirt-4, eroad, e-track-3, evo, mixed-1, road

# Deep Reinforcement Learning Project

## Install

- **Install required libraries**

sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng12-dev

- **Download repo**

git clone https://github.com/ugo-nama-kun/gym_torcs.git

- **cd into downloaded repo**

cd gym_torcs/vtorcs-RL-color/

./configure

- **below commands may require sudo**
make

make install

make datainstall


## Prepare

- **If you installed the game with sudo, you need to give your user account access to game files**

sudo chown -R YOUR_USERNAME /usr/local/share/games/torcs

- **Back to the main repo**

cd ..

- **Change the race configs**

mv quickrace.xml /usr/local/share/games/torcs/config/raceman/quickrace.xml


## How To Use

**TorcsEnv** is quite similar to a gym environment however there are a few things you need to keep in your mind.
- **Initialization**
  - While creating a TorcsEnv object you need to give path argument to use its full power. Path expects quickrace.xml file's path. **Example path:** "/usr/local/share/games/torcs/config/raceman/quickrace.xml"
- **Reset()**
  
  There are there arguments:
  - **relaunch**
    (default **False**) If true, relaunch the game. This is necessay if you changed the config or used any function that modifies it. Also, due to a memory leak you may want to relaunch the environment after a few normal resets(relaunch = False).
  - **sampletrack**
    (default **False**) If the path is given when the environment is initialized, randomly change the track.
  - **render**
    (default **True**) If given true modify the config file.

## Additional Notes

- You can modify reward and termination functions. However at the test time observations will be given in the same range. 
- Brakes can be disabled.
