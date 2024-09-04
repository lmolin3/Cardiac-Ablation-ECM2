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

if (MFEM_USE_MPI)

  list(APPEND ELECTROPHYIS_COMMON_SOURCES)
  list(APPEND ELECTROPHYIS_COMMON_HEADERS)

  convert_filenames_to_full_paths(ELECTROPHYIS_COMMON_SOURCES)
  convert_filenames_to_full_paths(ELECTROPHYIS_COMMON_HEADERS)

  set(ELECTROPHYIS_COMMON_FILES
      EXTRA_SOURCES ${ELECTROPHYIS_COMMON_SOURCES}
      EXTRA_HEADERS ${ELECTROPHYIS_COMMON_HEADERS})


  # Tests Bidomain
  add_mfem_miniapp(bidomain
  MAIN test_bidomain.cpp
  ${ELECTROPHYIS_COMMON_FILES}
  LIBRARIES mfem)

  # Create a symbolic link for meshes
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_CURRENT_SOURCE_DIR}/Mesh ${CMAKE_CURRENT_BINARY_DIR}/Mesh
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

endif ()
