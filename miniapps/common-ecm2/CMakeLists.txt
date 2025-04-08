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

mfem_add_library(mfem-common-ecm2 ${SRCS} ${HDRS})
target_link_libraries(mfem-common-ecm2 mfem)

# Check if we need to link stdc++fs for C++14
if (CMAKE_CXX_STANDARD LESS 17)
    target_link_libraries(mfem-common-ecm2 stdc++fs)
endif()

add_mfem_miniapp(MeshUtils
  MAIN MeshUtils.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common mfem-common-ecm2)

