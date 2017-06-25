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
----------------
```
./hello
```

Results
----------------
In a Radeon RX 470/480: Gallium 0.4 on AMD POLARIS10 (DRM 3.9.0 / 4.10.0-24-generic, LLVM 4.0.0)
```
Version: 4.5
Core profile: 1
time start (control):  hello.cpp:115
time end   (control): 0.003271  hello.cpp:117
time start (master):  hello.cpp:101
time end   (master): 0.088870  hello.cpp:103
time start (eevee):  hello.cpp:108
time end   (eevee): 0.189856  hello.cpp:110
```

That means Eevee shader compilation is taken 2x as much as master shaders.
And those ~200ms means there is a big lag every time the shader is recompiled (e.g., when the user drags a slider in a node).

Running from within Blender I get a similar result, so this sandbox seems well representative of the real production environment.

Note
----
Some systems create a cache for GLSL compiled shaders. So be sure to delete the cache after different runs of this program.

License
-------
The glsl shaders come from Blender (master, eevee) and Cycles (control). Licensed as GPL2 and Apache respectively.

Shaders
-------
The shaders come from Blender. They were obtained from the following patch in master and blender2.8.
The most basic shader (initial cube with a nodetree) was used for this test.
```
diff --git a/source/blender/gpu/intern/gpu_shader.c b/source/blender/gpu/intern/gpu_shader.c
index f0a1c182713..c9439f5df3f 100644
--- a/source/blender/gpu/intern/gpu_shader.c
+++ b/source/blender/gpu/intern/gpu_shader.c
@@ -267,6 +267,26 @@ GPUShader *GPU_shader_create(const char *vertexcode,
 	                            GPU_SHADER_FLAGS_NONE);
 }
 
+static void gpu_dump_shader(const char **code, const int num_shaders)
+{
+	const char *foldername = "/tmp/";
+	static int i = 0;
+
+	char filename[512] = {'\0'};
+	sprintf(filename, "%s/%04d.shader", foldername, i++);
+
+	FILE *f = fopen(filename, "w");
+	if (f == NULL) {
+		printf("Error writing to file: %s\n", filename);
+	}
+
+	for (int j = 0; j < num_shaders; j++) {
+		fprintf(f, "%s", code[j]);
+	}
+
+	fclose(f);
+}
+
 GPUShader *GPU_shader_create_ex(const char *vertexcode,
                                 const char *fragcode,
                                 const char *geocode,
@@ -325,6 +345,8 @@ GPUShader *GPU_shader_create_ex(const char *vertexcode,
 		if (defines) source[num_source++] = defines;
 		source[num_source++] = vertexcode;
 
+		gpu_dump_shader(source, num_source);
+
 		glAttachShader(shader->program, shader->vertex);
 		glShaderSource(shader->vertex, num_source, source, NULL);
 
@@ -364,6 +386,8 @@ GPUShader *GPU_shader_create_ex(const char *vertexcode,
 		if (libcode) source[num_source++] = libcode;
 		source[num_source++] = fragcode;
 
+		gpu_dump_shader(source, num_source);
+
 		glAttachShader(shader->program, shader->fragment);
 		glShaderSource(shader->fragment, num_source, source, NULL);
 
@@ -390,6 +414,8 @@ GPUShader *GPU_shader_create_ex(const char *vertexcode,
 		if (defines) source[num_source++] = defines;
 		source[num_source++] = geocode;
 
+		gpu_dump_shader(source, num_source);
+
 		glAttachShader(shader->program, shader->geometry);
 		glShaderSource(shader->geometry, num_source, source, NULL);
```
