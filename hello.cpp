#include "opengl.h"
#include <unistd.h>
#include <iostream>
#include <sstream>

#include "utils.h"
#include "shaders.h"

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
		const char *source;
		GLenum type;
	} shaders[2] = {
	    {
				.source = vertex_shader,
				.type = GL_VERTEX_SHADER,
			},
	    {
				.source = fragment_shader,
				.type = GL_FRAGMENT_SHADER,
			}
    };

	program = glCreateProgram();

	for(int i = 0; i < 2; i++) {
		GLuint shader = glCreateShader(shaders[i].type);

		// Read shader file into one string
		std::string shaderSource;
		if ( !readFileIntoString( shaders[i].source, shaderSource ) ) {
			return 0;
		}
		const GLchar *shader_source[] = { shaderSource.c_str() };

		glShaderSource(shader, 1, (const GLchar **)shader_source, NULL);
		glCompileShader(shader);

		glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

		if(!status) {
			glGetShaderInfoLog(shader, sizeof(log), &length, log);
			shader_print_errors("compile", log, shaderSource.c_str());
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
		//shader_print_errors("linking", log, shader_source);
		//shader_print_errors("linking", log, shader_source);
		return 0;
	}

	return program;
}

static void bind_master_shader(void)
{
	TIMEIT_START(master);
	bind_shader("shaders/master.fp", "shaders/master.vp");
	TIMEIT_END(master);
};

static void bind_eevee_shader(void)
{
	TIMEIT_START(eevee);
	bind_shader("shaders/eevee.fp", "shaders/eevee.vp");
	TIMEIT_END(eevee);
};

static void bind_fallback_shader(void)
{
	TIMEIT_START(control);
	bind_shader("shaders/control.fp", "shaders/control.vp");
	TIMEIT_END(control);
}

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
	Window win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, 100, 100, 0, vi->depth, InputOutput, vi->visual, CWBorderPixel|CWColormap|CWEventMask, &swa);
 
	XMapWindow (dpy, win);
 
	//oldstyle context:
	//	GLXContext ctx = glXCreateContext(dpy, vi, 0, GL_TRUE);
 
	std::cout << "glXCreateContextAttribsARB " << (void*) glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB") << std::endl;
	GLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = (GLXCREATECONTEXTATTRIBSARBPROC) glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");
 
	int attribs[] = {
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
		GLX_CONTEXT_MINOR_VERSION_ARB, 3,
		0};
 
	GLXContext ctx = glXCreateContextAttribsARB(dpy, *fbc, 0, true, attribs);
 
	glXMakeCurrent (dpy, win, ctx);

	int major, minor, mask;
	int error;

	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);
	glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);

	std::cout << "Version: " << major << "." << minor << std::endl;
	std::cout << "Core profile: " << (mask & GLX_CONTEXT_CORE_PROFILE_BIT_ARB) << std::endl;
 
	glClearColor (0, 0.5, 1, 1);
	glClear (GL_COLOR_BUFFER_BIT);
	glXSwapBuffers (dpy, win);

	/* cleanup the error stack */
	glGetError();

	//TIMEIT_START(shader_time);

	bind_fallback_shader();
	//TIMEIT_VALUE_PRINT(shader_time);

	bind_master_shader();
	//TIMEIT_VALUE_PRINT(shader_time);

	bind_eevee_shader();
	//TIMEIT_END(shader_time);

	//glColor4f(1.0, 0.0, 0.0, 1.0);
	//glLineStipple(3, 0xAAAA);
	GLint pro;
	//glGetIntegerv(GL_CURRENT_PROGRAM, &pro);
	glGetIntegerv(GL_ACTIVE_PROGRAM, &pro);
	error = glGetError();
	std::cout << "Error ? " << error << " " << GL_INVALID_OPERATION <<
			" " << GL_INVALID_ENUM
			<< std::endl;

	glClearColor (2, 0.5, 0, 1);
	glClear (GL_COLOR_BUFFER_BIT);
	glXSwapBuffers (dpy, win);
 
	ctx = glXGetCurrentContext(); 
	glXDestroyContext(dpy, ctx); 
	}
