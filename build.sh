#!/bin/bash

osx_version="10.13"
export MACOSX_DEPLOYMENT_TARGET=$osx_version

maxsdk="./include/max-sdk-7.3.3/source/c74support"
frameworks='-F./include/Frameworks -framework MaxAudioAPI -framework MaxAPI -framework Carbon'
warnings="-Werror -W -Wall -Wextra -Wno-unused-parameter -Wno-reorder -Wno-sign-compare -Wno-unused-function -Wno-unused-label -Wno-unknown-pragmas -Wno-unused-value -Wno-unused-variable"
dylib="-Xlinker -dylib"
bundle="-Xlinker -bundle"
include="-I$maxsdk/max-includes -I$maxsdk/msp-includes"
common="-Wl,-undefined,dynamic_lookup $include -fvisibility=hidden -mmacosx-version-min=$osx_version -D MAC_VERSION $warnings"
debug="-DDEBUG=1 -O0 -g"
release="-DDEBUG=0 -DNDEBUG=1 -O3"
standalone="-DSTANDALONE=1"

ispc fft.ispc -O3 -o fft_ispc.o -h fft_ispc.h --opt=fast-masked-vload --opt=fast-math --target=avx2-i32x8 --math-lib=fast --wno-perf
clang $standalone $debug $common ./dsp.c fft_ispc.o -o ./build/dsp
clang $debug $bundle $common $frameworks ./sparse-interp~.c fft_ispc.o -o ./build/sparse-interp~.mxo/Contents/MacOS/sparse-interp~