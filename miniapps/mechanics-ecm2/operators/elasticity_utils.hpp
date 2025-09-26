#pragma once
#include "mfem.hpp"

namespace mfem
{
    namespace elasticity_ecm2
    {

        using VecFuncT = void(const Vector &x, real_t t, Vector &u);
        using ScalarFuncT = real_t(const Vector &x, real_t t);

        class FixedConstraint
        {
        public:
            FixedConstraint(const Array<int> &attr, int component_ = -1) : attr(attr), component(component_) {}

            FixedConstraint(const FixedConstraint &other)
                : attr(other.attr), component(other.component) {}

            FixedConstraint(FixedConstraint &&other) noexcept
                : attr(std::move(other.attr)), component(other.component) {}

            Array<int> attr; ///< Array marking the boundary attributes for fixed constraints.
            int component;   ///< Component of the field to which the fixed boundary condition applies (-1 for all components).
        };

    } // namespace elasticity_ecm2
} // namespace mfem

