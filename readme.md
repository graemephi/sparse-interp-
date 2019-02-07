# sparse-interp~

Running requires Ableton 10 (lower might work but not tested) with Max for Live.

Building requires a C compiler (Visual Studio on windows or clang on macos), Intel's ispc, and the max SDK. Place version 7.3.3 of the max SDK in ./include.

Building on windows:

    Run build.bat from VS Tools command prompt.

Building on macos:

    Run build.sh

Place the resulting .mxe64 or .mxo somewhere Max can see it and run the amxd in ./amxd.