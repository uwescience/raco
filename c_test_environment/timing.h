#pragma once


#if defined(__MTA__)
#include <sys/mta_task.h>
#include <machine/runtime.h>
#elif defined(__MACH__)
#include <mach/mach_time.h>
#else
#include <time.h>
#endif


#define BILLION 1000000000


/// "Universal" wallclock time (works at least for Mac, MTA, and most Linux)
inline double walltime(void) {
#if defined(__MTA__)
        return((double)mta_get_clock(0) / mta_clock_freq());
#elif defined(__MACH__)
        static mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        uint64_t now = mach_absolute_time();
        now *= info.numer;
        now /= info.denom;
        return 1.0e-9 * (double)now;
#else
        struct timespec tp;
#if defined(CLOCK_PROCESS_CPUTIME_ID)
#define CLKID CLOCK_PROCESS_CPUTIME_ID
#elif  defined(CLOCK_REALTIME_ID)
#define CLKID CLOCK_REALTIME_ID
#endif
        clock_gettime(CLOCK_MONOTONIC, &tp);
        return (double)tp.tv_sec + (double)tp.tv_nsec / BILLION;
#endif
}
