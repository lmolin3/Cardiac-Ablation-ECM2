# Parallel apps.
if (MFEM_USE_MPI)

  add_mfem_miniapp(test-transfer-two-domains-overlapping-partition
    MAIN test-transfer-two-domains-overlapping-partition.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem mfem-common-ecm2)

  add_mfem_miniapp(test-nonoverlapping-partition
    MAIN test-nonoverlapping-partition.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem)
      
endif()

