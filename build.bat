@echo off

REM VSCodes no good sometimes
if "%1" == "clear" (
    clear
    clear
)

REM call build_dsp.bat
REM goto end

set BaseInclude=..\include
set MaxSDK=%BaseInclude%\max-sdk-7.3.3
set MaxInclude=%MaxSDK%\source\c74support\max-includes
set MSPInclude=%MaxSDK%\source\c74support\msp-includes
set MaxLibs=%MSPInclude%\x64\MaxAudio.lib %MaxInclude%\x64\MaxAPI.lib

set LinkerOptions=-EXPORT:ext_main -INCREMENTAL:no fft_ispc.o

set Debug=-Od -Ob1 -LD
set Release=-Ox -Ob2 -LDd
set CompilerOptions=-Zi -Oi -Gy %Release% -W3 -WX -nologo -fp:fast -DMAXAPI_USE_MSCRT -DWIN_VERSION -DEXT_WIN_VERSION

mkdir build 2> nul
pushd build
ispc ..\fft.ispc -O3 -o fft_ispc.o -h ..\fft_ispc.h --opt=fast-masked-vload --opt=fast-math --target=sse2-i32x8 --math-lib=fast --wno-perf
call cl.exe %CompilerOptions% -I%MaxInclude% -I%MSPInclude% -Fe:sparse-interp~.mxe64 ..\sparse-interp~.c %MaxLibs% -link %LinkerOptions%
popd

:end