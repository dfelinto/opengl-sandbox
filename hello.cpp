#include <unistd.h>
#include <iostream>
#include <sstream>

#include "opengl.h"
#include "utils.h"

#define DEBUG_VERBOSE

#ifndef DEBUG_VERBOSE
  #define VERBOSE_TIMEIT_START(x)
  #define VERBOSE_TIMEIT_VALUE_PRINT(x)
  #define VERBOSE_TIMEIT_END(x)
#else
  #define VERBOSE_TIMEIT_START TIMEIT_START
  #define VERBOSE_TIMEIT_VALUE_PRINT TIMEIT_VALUE_PRINT
  #define VERBOSE_TIMEIT_END TIMEIT_END
#endif

#define GLX_CONTEXT_MAJOR_VERSION_ARB		0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB		0x2092
typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display*, GLXFBConfig, GLXContext, Bool, const int*);


static void shader_print_errors(const char *task, const char *log, const char *code)
{
	std::cerr << "Shader: " << task << " error:" << std::endl;
	std::cerr << "===== shader string ====" << std::endl;

	std::stringstream stream(code);
	std::string partial;

	int line = 1;
	while (getline(stream, partial, '\n')) {
		if (line < 10) {
			std::cerr << " " << line << " " << partial << std::endl;
		}
		else {
			std::cerr << line << " " << partial << std::endl;
		}
		line++;
	}
	std::cerr << log << std::endl;
}

static int bind_shader(const char *fragment_shader, const char *vertex_shader)
{
	GLint status;
	GLchar log[5000];
	GLsizei length = 0;
	GLuint program = 0;

	struct Shader {
		const char *filename;
		std::string source;
		GLenum type;
	} shaders[2] = {
	    {
				.filename = vertex_shader,
				.source = "",
				.type = GL_VERTEX_SHADER,
			},
	    {
				.filename = fragment_shader,
				.source = "",
				.type = GL_FRAGMENT_SHADER,
			}
    };

	program = glCreateProgram();

	for (int i = 0; i < 2; i++) {

		GLuint shader = glCreateShader(shaders[i].type);

		/* Read shader file into one string. */
		if (!readFileIntoString(shaders[i].filename, shaders[i].source)) {
			return 0;
		}
		const GLchar *shader_source[] = { shaders[i].source.c_str() };

		glShaderSource(shader, 1, (const GLchar **)shader_source, NULL);
		glCompileShader(shader);

		glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

		if(!status) {
			glGetShaderInfoLog(shader, sizeof(log), &length, log);
			shader_print_errors("compile", log, shaders[i].source.c_str());
			return 0;
		}

		glAttachShader(program, shader);
	}

	/* Link output. */
	glBindFragDataLocation(program, 0, "fragColor");

	/* Link and error check. */
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &status);

	if(!status) {
		glGetShaderInfoLog(program, sizeof(log), &length, log);
		std::cerr << "Linking error" << std::endl;
		shader_print_errors("linking", log, shaders[0].source.c_str());
		shader_print_errors("linking", log, shaders[1].source.c_str());
		return 0;
	}

	return program;
}

static int compile_shader_init(void)
{
	return bind_shader("shaders/init.fp", "shaders/fullscreen.vp");
};

static int compile_shader_display(void)
{
	return bind_shader("shaders/display.fp", "shaders/fullscreen.vp");
};

static int compile_shader_downsample(void)
{
	return bind_shader("shaders/downsample.fp", "shaders/fullscreen.vp");
};

int main (int argc, char ** argv){
	Display *dpy = XOpenDisplay(0);
 
	int nelements;
	GLXFBConfig *fbc = glXChooseFBConfig(dpy, DefaultScreen(dpy), 0, &nelements);
 
	static int attributeList[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1, GLX_BLUE_SIZE, 1, None };
	XVisualInfo *vi = glXChooseVisual(dpy, DefaultScreen(dpy),attributeList);
 
	XSetWindowAttributes swa;
	swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
	swa.border_pixel = 0;
	swa.event_mask = StructureNotifyMask;
	Window win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, 512, 512, 0, vi->depth, InputOutput, vi->visual, CWBorderPixel|CWColormap|CWEventMask, &swa);
 
	XMapWindow(dpy, win);
 
	GLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = (GLXCREATECONTEXTATTRIBSARBPROC) glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");
 
	int attribs[] = {
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
		GLX_CONTEXT_MINOR_VERSION_ARB, 3,
		0};
 
	GLXContext ctx = glXCreateContextAttribsARB(dpy, *fbc, 0, true, attribs);
 
	glXMakeCurrent(dpy, win, ctx);

	int major, minor, mask;
	int error;

	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);
	glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);

	const GLubyte* vendor = glGetString(GL_VENDOR);
	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* version = glGetString(GL_VERSION);

	std::cout << "Version: " << major << "." << minor << std::endl;
	std::cout << "Core profile: " << (mask & GLX_CONTEXT_CORE_PROFILE_BIT_ARB) << std::endl;

	std::cout << "Vendor: " << vendor << std::endl;
	std::cout << "Renderer: " << renderer << std::endl;
	std::cout << "Version: " << version << std::endl;
 
	// -------- Fullscreen tri ----------- //
    float vertices[] = {
	     -1.0f, -1.0f,
	     -1.0f,  3.0f,
	      3.0f, -1.0f,
    };
    unsigned int vbo, vao;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // -------- Texture ----------- //
    unsigned int tex;
    unsigned int tex_size = 64;
	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R11F_G11F_B10F, tex_size, tex_size, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 2);
	glGenerateMipmap(GL_TEXTURE_2D);

	// --------- Framebuffers ---------- //
    unsigned int fbo;
	glGenFramebuffers(1, &fbo);

	// ---------- Shaders --------- //
	int init_sh = compile_shader_init();
	int display_sh = compile_shader_display();
	int downsample_sh = compile_shader_downsample();

	// - WAIT - //
	sleep(1);

	// -------- DRAW --------------//
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glDisable(GL_DEPTH_TEST);

	glUseProgram(init_sh);

    // MIP 0
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
	glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

	glUseProgram(downsample_sh);
	glUniform1i(0, 0);

    // MIP 1
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 1);
	glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // MIP 2
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 2);
	glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Reset
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 2);

    // ---------- DRAW TO OUTPUT ---------- //
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glDisable(GL_DEPTH_TEST);
	glUseProgram(display_sh);
	glUniform1i(0, 0);
	glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);


	/* cleanup the error stack */
	if (glGetError()) {
		printf("ERROR\n");
	};

	glXSwapBuffers(dpy, win);
 
	ctx = glXGetCurrentContext();
	sleep(5);	
	glXDestroyContext(dpy, ctx); 
	}
