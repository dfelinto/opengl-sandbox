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

Rendering to individual mipmap seems fine as this apitrace screenshot suggests. However the sampling of any mip equal or smaller than 16x16 pixels is giving garbage output. This does not appear if texture is GL_RGBA32F.
