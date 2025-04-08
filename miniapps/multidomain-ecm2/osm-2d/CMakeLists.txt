# Parallel apps.
if (MFEM_USE_MPI)

  add_mfem_miniapp(multidomain-two-domains-rf-osm-2d
    MAIN multidomain-two-domains-rf.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 electrostatics_ecm2)  

endif()

