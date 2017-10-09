OpenGL SandBox
==============

Getting a sandbox-like environment for OpenGL debugging.
For now I'm profiling shaders in *master* and *blender2.8* branches of Blender.
Originally I was using this for testing the setup of OpenGL core profile.

How to build it?
----------------
```
make
```

How to run it?
--------------
```
./hello
```

Expected results
----------------
![Working image](http://www.dalaifelinto.com/ftp/opengl-vega-working.png)

Buggy results
-------------
![Buggy image](http://www.dalaifelinto.com/ftp/opengl-vega-bug.png)

Details
-------
![Buggy image](http://www.dalaifelinto.com/ftp/opengl-vega-details.png)

Here is the bug visually explained. The issue starts when mipmap is 16 or lower.
