#ifndef __SHADERS__
#define __SHADERS__

const char *FALLBACK_VERTEX_SHADER =
"#version 330\n"
"uniform vec2 fullscreen;\n"
"in vec2 texCoord;\n"
"in vec2 pos;\n"
"out vec2 texCoord_interp;\n"
"\n"
"vec2 normalize_coordinates()\n"
"{\n"
"	return (vec2(2.0) * (pos / fullscreen)) - vec2(1.0);\n"
"}\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(normalize_coordinates(), 0.0, 1.0);\n"
"	texCoord_interp = texCoord;\n"
"}\n\0";

const char *FALLBACK_FRAGMENT_SHADER =
"#version 330\n"
"uniform sampler2D image_texture;\n"
"in vec2 texCoord_interp;\n"
"out vec4 fragColor;\n"
"\n"
"void main()\n"
"{\n"
"	fragColor = texture(image_texture, texCoord_interp);\n"
"}\n\0";

#endif /* __SHADERS__ */
