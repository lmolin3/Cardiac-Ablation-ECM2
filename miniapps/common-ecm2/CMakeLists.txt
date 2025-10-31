# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.


list(APPEND SRCS
  utils.cpp
  interface_transfer.cpp
  )
list(APPEND HDRS
  utils.hpp
  interface_transfer.hpp
  )

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(MINIAPPS_COMMON_SOURCES ${MINIAPPS_COMMON_SOURCES} ${SRCS} PARENT_SCOPE)
set(MINIAPPS_COMMON_HEADERS ${MINIAPPS_COMMON_HEADERS} ${HDRS} PARENT_SCOPE)

if (MFEM_USE_CUDA)
  set_property(SOURCE ${SRCS} PROPERTY LANGUAGE CUDA)
endif()

mfem_add_library(mfem-common-ecm2 ${SRCS} ${HDRS})
target_link_libraries(mfem-common-ecm2 mfem)

# Check if we need to link stdc++fs for C++14
if (CMAKE_CXX_STANDARD LESS 17)
    target_link_libraries(mfem-common-ecm2 stdc++fs)
endif()

# Add executable for Mesh Utilities
add_mfem_miniapp(mesh-utils
  MAIN mesh-utils.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common mfem-common-ecm2)

# Add tests for the interface transfer class
add_mfem_miniapp(transfer2d
  MAIN tests/transfer2d.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common mfem-common-ecm2)

add_mfem_miniapp(transfer3d
  MAIN tests/transfer3d.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common mfem-common-ecm2)

# Copy mesh data files
set(MESH_GF_FILES
    multidomain2d-tri.e
    multidomain2d-quad.e
)
add_custom_command(OUTPUT data_is_copied
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MESH_GF_FILES} .
  COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
  COMMENT "Copying common-fsi miniapps data files ...")
add_custom_target(copy_miniapps_common_fsi_data DEPENDS data_is_copied)