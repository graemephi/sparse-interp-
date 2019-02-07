@echo off

REM VSCodes no good sometimes
if "%1" == "clear" (
    clear
    clear
)

set DLLDebug=-Od -Ob1 -LDd
set DLLRelease=-Ox -Ob2 -LD
set EXEDebug=-Od -Ob1 -MTd -DSTANDALONE=1
set EXERelease=-Ox -Ob2 -MT -DSTANDALONE=1
set CompilerOptions=-Zi -Oi -Gy -W4 -WX -nologo -fp:fast

REM ispc fft.ispc -O3 -o fft_ispc.o -h fft_ispc.h --opt=fast-masked-vload --opt=fast-math --target=avx2-i32x8 --math-lib=fast --wno-perf

mkdir build 2> nul
pushd build
    call cl.exe %CompilerOptions% %DLLRelease% -Fe:dsp.dll ..\dsp.c -link -INCREMENTAL:no ..\fft_ispc.o
    REM call cl.exe %CompilerOptions% %DLLRelease% -FAu -Fa:dsp.asm ..\dsp.c -link -INCREMENTAL:no ..\fft_ispc.o
    mkdir exe_test 2> nul
    pushd exe_test
        call cl.exe %CompilerOptions% %EXEDebug% -Fe:dsp.exe ..\..\dsp.c -link -INCREMENTAL:no ..\..\fft_ispc.o
    popd
popd

:end