# RCSnailAI
Student project for the Neural Networks course.

* GOAL: Using neural networks for controlling RC car on a specific race track. Input is based only on camera feed.


Our approach is to solve it as regression problem with 4 unknown variables which are throttle [0-1], braking [0-1], steering [-1 - 1] and gear [0,1,2] (Reverse, neutral, drive). Input is given as video feed which must be separated to frames and solve the problem for each frame separately. 
