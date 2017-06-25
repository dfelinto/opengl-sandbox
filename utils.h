#ifndef __UTILS__
#define __UTILS__

#include <unistd.h>
#include <sys/time.h>

double PIL_check_seconds_timer(void) 
{
	struct timeval tv;
	struct timezone tz;

	gettimeofday(&tv, &tz);

	return ((double) tv.tv_sec + tv.tv_usec / 1000000.0);
}

long int PIL_check_seconds_timer_i(void)
{
	struct timeval tv;
	struct timezone tz;

	gettimeofday(&tv, &tz);

	return tv.tv_sec;
}

void PIL_sleep_ms(int ms)
{
	if (ms >= 1000) {
		sleep(ms / 1000);
		ms = (ms % 1000);
	}
	
	usleep(ms * 1000);
}

#define STRINGIFY_APPEND(a, b) "" a #b
#define STRINGIFY(x) STRINGIFY_APPEND("", x)
#define AT __FILE__ ":" STRINGIFY(__LINE__)

#define TIMEIT_START(var)                                                 \
	{                                                                       \
		double _timeit_##var = PIL_check_seconds_timer();                     \
		printf("time start (" #var "):  " AT "\n");                           \
		fflush(stdout);                                                       \
		{ (void)0

/**
 * \return the time since TIMEIT_START was called.
 */
#define TIMEIT_VALUE(var) (float)(PIL_check_seconds_timer() - _timeit_##var)

#define TIMEIT_VALUE_PRINT(var)                                               \
	{                                                                         \
		printf("time update   (" #var "): %.6f" "  " AT "\n", TIMEIT_VALUE(var));\
		fflush(stdout);                                                       \
	} (void)0

#define TIMEIT_END(var)                                                       \
		}                                                                     \
		printf("time end   (" #var "): %.6f" "  " AT "\n", TIMEIT_VALUE(var));\
		fflush(stdout);                                                       \
	} (void)0



#endif /* __UTILS__ */
