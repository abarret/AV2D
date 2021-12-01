## ---------------------------------------------------------------------
##
## Copyright (c) 2020 - 2020 by the IBAMR developers
## All rights reserved.
##
## This file is part of IBAMR.
##
## IBAMR is free software and is distributed under the 3-clause BSD
## license. The full text of the license can be found in the file
## COPYRIGHT at the top level directory of IBAMR.
##
## ---------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

PROJECT(AV_2D LANGUAGES C CXX Fortran)
FIND_PACKAGE(ADS REQUIRED)

SET(CXX_SRC
    CBFinder.cpp
    CirculationModel.cpp
    FeedbackForcer.cpp
    QFcn.cpp
    VelocityBcCoefs.cpp 
    RBFReconstructCacheOS.cpp 
    RBFOneSidedReconstructions.cpp
    )
ADD_EXECUTABLE(main2d main.cpp)
TARGET_SOURCES(main2d PRIVATE ${CXX_SRC})
TARGET_LINK_LIBRARIES(main2d ADS::ADS2d)
TARGET_LINK_LIBRARIES(main2d IBAMR::IBAMR2d)
CONFIGURE_FILE(input2d ${CMAKE_CURRENT_BINARY_DIR}/input2d COPYONLY)
CONFIGURE_FILE(2d_aorta_sideset.e ${CMAKE_CURRENT_BINARY_DIR}/2d_aorta_sideset.e COPYONLY)
CONFIGURE_FILE(2d_leaflets_sideset.e ${CMAKE_CURRENT_BINARY_DIR}/2d_leaflets_sideset.e COPYONLY)