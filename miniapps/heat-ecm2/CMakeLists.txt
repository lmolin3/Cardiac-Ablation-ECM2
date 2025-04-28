if (MFEM_USE_MPI)

  list(APPEND HEAT_COMMON_SOURCES
    lib/heat_solver.cpp
    lib/heat_operator.cpp
    lib/implicit_solver.cpp
    bc/heat_bchandler.cpp
  )

  list(APPEND HEAT_COMMON_HEADERS
    lib/heat_solver.hpp
    lib/heat_operator.hpp
    lib/implicit_solver.hpp
    bc/heat_bchandler.hpp
  )

  convert_filenames_to_full_paths(HEAT_COMMON_SOURCES)
  convert_filenames_to_full_paths(HEAT_COMMON_HEADERS)

  set(HEAT_COMMON_FILES
      EXTRA_SOURCES ${HEAT_COMMON_SOURCES}
      EXTRA_HEADERS ${HEAT_COMMON_HEADERS})

  mfem_add_library(heat-ecm2 ${HEAT_COMMON_SOURCES} ${HEAT_COMMON_HEADERS})
  target_link_libraries(heat-ecm2 mfem mfem-common-ecm2 mfem-common)

  add_mfem_miniapp(heat_test
    MAIN tests/heat_test.cpp
    ${HEAT_COMMON_FILES}
    LIBRARIES heat-ecm2)



  add_mfem_miniapp(convergence_heat
    MAIN tests/convergence.cpp
    ${HEAT_COMMON_FILES}
    LIBRARIES heat-ecm2)

  add_mfem_miniapp(heat_test_3Daniso
    MAIN tests/heat_test_3Daniso.cpp
    ${HEAT_COMMON_FILES}
    LIBRARIES heat-ecm2)

endif () #MFEM_USE_MPI

