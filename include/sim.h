#ifndef __SIM_API
#define __SIM_API

#define SIM_CMD_ROI_TOGGLE      0
#define SIM_CMD_ROI_START       1
#define SIM_CMD_ROI_END         2
#define SIM_CMD_MHZ_SET         3
#define SIM_CMD_MARKER          4
#define SIM_CMD_USER            5
#define SIM_CMD_INSTRUMENT_MODE 6
#define SIM_CMD_MHZ_GET         7
#define SIM_CMD_IN_SIMULATOR    8
#define SIM_CMD_PROC_ID         9
#define SIM_CMD_THREAD_ID       10
#define SIM_CMD_NUM_PROCS       11
#define SIM_CMD_NUM_THREADS     12
#define SIM_CMD_NAMED_MARKER    13
#define SIM_CMD_SET_THREAD_NAME 14

#define SIM_OPT_INSTRUMENT_DETAILED    0
#define SIM_OPT_INSTRUMENT_WARMUP      1
#define SIM_OPT_INSTRUMENT_FASTFORWARD 2

#if defined(__aarch64__)

inline unsigned long SimMagic0(unsigned long cmd) {
    unsigned long res;
    asm volatile (
        "mov x1, %[x]\n"
        "\tbfm x0, x0, 0, 0\n"
        : [ret]"=r"(res)
        : [x]"r"(cmd)
    );
    return res;
}

inline unsigned long SimMagic1(unsigned long cmd, unsigned long arg0) {
    unsigned long res;
    asm volatile (
        "mov x1, %[x]\n"
        "\tmov x2, %[y]\n"
        "\tbfm x0, x0, 0, 0\n"
        : [ret]"=r"(res)
        : [x]"r"(cmd),
          [y]"r"(arg0)
        : "x2", "x1"
    );
    return res;
}

inline unsigned long SimMagic2(unsigned long cmd, unsigned long arg0, unsigned long arg1) {
    unsigned long res;
    asm volatile (
        "mov x1, %[x]\n"
        "\tmov x2, %[y]\n"
        "\tmov x3, %[z]\n"
        "\tbfm x0, x0, 0, 0\n"
        : [ret]"=r"(res)
        : [x]"r"(cmd),
          [y]"r"(arg0),
          [z]"r"(arg1)
        : "x1", "x2", "x3"
    );
    return res;
}

#else  // x86 architecture

#if defined(__i386)
    #define MAGIC_REG_A "eax"
    #define MAGIC_REG_B "edx" // Required for -fPIC support
    #define MAGIC_REG_C "ecx"
#else
    #define MAGIC_REG_A "rax"
    #define MAGIC_REG_B "rbx"
    #define MAGIC_REG_C "rcx"
#endif

inline unsigned long SimMagic0(unsigned long cmd) {
    unsigned long res;
    __asm__ __volatile__ (
        "mov %1, %%" MAGIC_REG_A "\n"
        "\txchg %%bx, %%bx\n"
        : "=a" (res)
        : "g"(cmd)
    );
    return res;
}

inline unsigned long SimMagic1(unsigned long cmd, unsigned long arg0) {
    unsigned long res;
    __asm__ __volatile__ (
        "mov %1, %%" MAGIC_REG_A "\n"
        "\tmov %2, %%" MAGIC_REG_B "\n"
        "\txchg %%bx, %%bx\n"
        : "=a" (res)
        : "g"(cmd),
          "g"(arg0)
        : "%" MAGIC_REG_B
    );
    return res;
}

inline unsigned long SimMagic2(unsigned long cmd, unsigned long arg0, unsigned long arg1) {
    unsigned long res;
    __asm__ __volatile__ (
        "mov %1, %%" MAGIC_REG_A "\n"
        "\tmov %2, %%" MAGIC_REG_B "\n"
        "\tmov %3, %%" MAGIC_REG_C "\n"
        "\txchg %%bx, %%bx\n"
        : "=a" (res)
        : "g"(cmd),
          "g"(arg0),
          "g"(arg1)
        : "%" MAGIC_REG_B, "%" MAGIC_REG_C
    );
    return res;
}

#endif

// Standard C++ inline functions instead of macros
inline void SimRoiStart() { SimMagic0(SIM_CMD_ROI_START); }
inline void SimRoiEnd() { SimMagic0(SIM_CMD_ROI_END); }
inline unsigned long SimGetProcId() { return SimMagic0(SIM_CMD_PROC_ID); }
inline unsigned long SimGetThreadId() { return SimMagic0(SIM_CMD_THREAD_ID); }
inline void SimSetThreadName(const char* name) { SimMagic1(SIM_CMD_SET_THREAD_NAME, (unsigned long)(name)); }
inline unsigned long SimGetNumProcs() { return SimMagic0(SIM_CMD_NUM_PROCS); }
inline unsigned long SimGetNumThreads() { return SimMagic0(SIM_CMD_NUM_THREADS); }
inline void SimSetFreqMHz(unsigned long proc, unsigned long mhz) { SimMagic2(SIM_CMD_MHZ_SET, proc, mhz); }
inline void SimSetOwnFreqMHz(unsigned long mhz) { SimSetFreqMHz(SimGetProcId(), mhz); }
inline unsigned long SimGetFreqMHz(unsigned long proc) { return SimMagic1(SIM_CMD_MHZ_GET, proc); }
inline unsigned long SimGetOwnFreqMHz() { return SimGetFreqMHz(SimGetProcId()); }
inline void SimMarker(unsigned long arg0, unsigned long arg1) { SimMagic2(SIM_CMD_MARKER, arg0, arg1); }
inline void SimNamedMarker(unsigned long arg0, const char* str) { SimMagic2(SIM_CMD_NAMED_MARKER, arg0, (unsigned long)(str)); }
inline void SimUser(unsigned long cmd, unsigned long arg) { SimMagic2(SIM_CMD_USER, cmd, arg); }
inline void SimSetInstrumentMode(unsigned long opt) { SimMagic1(SIM_CMD_INSTRUMENT_MODE, opt); }
inline bool SimInSimulator() { return (SimMagic0(SIM_CMD_IN_SIMULATOR) != SIM_CMD_IN_SIMULATOR); }

#endif /* __SIM_API */