// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#pragma once

#ifndef NAVIER_TYPES_HPP
#define NAVIER_TYPES_HPP

namespace mfem
{
    namespace navier
    {
        enum class TimeAdaptivityType : int
        {
            NONE = 0, // Fixed time step
            CFL = 1,  // CFL-based adaptivity
            HOPC = 2  // High Order Pressure Correction based adaptivity
        };

        enum class BlockPreconditionerType : int
        {
            BLOCK_DIAGONAL = 0,
            LOWER_TRIANGULAR = 1,
            UPPER_TRIANGULAR = 2,
            CHORIN_TEMAM = 3,
            YOSIDA = 4,
            CHORIN_TEMAM_PRESSURE_CORRECTED = 5,
            YOSIDA_PRESSURE_CORRECTED = 6,
            YOSIDA_HIGH_ORDER_PRESSURE_CORRECTED = 7
        };

        enum class SchurPreconditionerType : int
        {
            PRESSURE_MASS = 0,
            PRESSURE_LAPLACIAN = 1,
            PCD = 2,
            CAHOUET_CHABARD = 3,
            LSC = 4,
            APPROXIMATE_DISCRETE_LAPLACIAN = 5
        };

    }
}

#endif // NAVIER_TYPES_HPP