This is a two dimensional model of flow through the aortic valve. A concentration field is advected and diffused along with the flow. As it touches the reaction surface, it can bind and unbind to the surface, causing that portion of the leaflet to become stiffer.

Requires a build of CutCellAdvDiff, which requires a CMake build of IBAMR. This example can be configured using

cmake -DCMAKE_C_FLAGS="-O3 -march=native" -DCMAKE_CXX_FLAGS="-O3 -march=native" -DCMAKE_Fortran_FLAGS="-O3 -march=native" -DADS_ROOT=/path/to/ads/build ../AV2D

Sometimes compilers must be given to cmake.
