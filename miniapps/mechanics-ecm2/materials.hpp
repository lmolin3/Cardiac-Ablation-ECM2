#pragma once

#include "mfem.hpp"
#include <variant>


using namespace mfem;
using namespace mfem::future;

namespace mfem
{
   namespace elasticity_ecm2
   {

      // To add a new material:
      // 1. Define a new struct inheriting from MaterialConcept<dim>
      // 2. Implement the operator() method and static constexpr name (this computes the 1st Piola-Kirchhoff stress given dudX)
      // 3. Create the factory function for creating the material (and passing parameters)
      // 4. Add the new material to the MaterialVariant type alias in ElasticityOperator.hpp

      // Base concept for all materials - helps with type safety
      template <int dim>
      struct MaterialConcept
      {
         // This is just a concept - no runtime polymorphic base class,
         // but a compile-time interface check, for performance.
         // Each material must implement:
         // auto operator()(const tensor<real_t, dim, dim> &dudX) const
         // static constexpr const char* name
      };


      // Linear elastic material (for small strains)
      template <int dim>
      struct LinearElastic : public MaterialConcept<dim>
      {
         static constexpr const char *name = "LinearElastic";  
         MFEM_HOST_DEVICE inline auto operator()(const tensor<real_t, dim, dim> &dudX) const
         {
            constexpr auto I = IsotropicIdentity<dim>();
            const real_t mu = E / (2.0 * (1.0 + nu));
            const real_t lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

            // Small strain tensor
            auto strain = 0.5 * (dudX + transpose(dudX));
            auto stress = 2.0 * mu * strain + lambda * tr(strain) * I;

            // First Piola-Kirchhoff stress (approximation for small strains) P \approx= sigma
            return stress;
         }

         real_t nu = 0.3; // Poisson's ratio
         real_t E = 1.0e6;  // Young's modulus
      };

      // Saint-Venant Kirchhoff
      template <int dim>
      struct SaintVenantKirchoff : public MaterialConcept<dim>
      {
         static constexpr const char *name = "SaintVenantKirchoff";

         MFEM_HOST_DEVICE inline auto operator()(const tensor<real_t, dim, dim> &dudX) const
         {
            constexpr auto I = IsotropicIdentity<dim>();
            const real_t mu = E / (2.0 * (1.0 + nu));
            const real_t lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

            // Green-Lagrange strain
            auto strain = 0.5 * (dudX + transpose(dudX) + transpose(dudX) * dudX);
            auto stress = 2.0 * mu * strain + lambda * tr(strain) * I;

            // Convert to first Piola-Kirchhoff
            return (I + dudX) * stress;
         }

         real_t nu = 0.3; // Poisson's ratio
         real_t E = 1.0e6;  // Young's modulus
      };


      // Neo-Hookean material
      template <int dim>
      struct NeoHookean : public MaterialConcept<dim>
      {
         static constexpr const char *name = "NeoHookean";

         MFEM_HOST_DEVICE inline auto operator()(const tensor<real_t, dim, dim> &dudX) const
         {
            constexpr auto I = IsotropicIdentity<dim>();

            // Neo-Hookean: W = (mu/2)(I1 - 3) - mu*ln(J) + (kappa/2)(ln(J))^2
            auto F = dudX + I;
            auto J = det(F);
            auto logJ = log(J);
            return mu * F + (kappa * logJ - mu) * inv(transpose(F));
         }

         real_t kappa = 1.0e6;  // Bulk modulus
         real_t mu = 1.0e3;     // Shear modulus
      };

      // Mooney-Rivlin material with two material constants
      template <int dim>
      struct MooneyRivlin : public MaterialConcept<dim>
      {
         static constexpr const char *name = "MooneyRivlin";

         MFEM_HOST_DEVICE inline auto operator()(const tensor<real_t, dim, dim> &dudX) const
         {
            constexpr auto I = IsotropicIdentity<dim>();
            auto F = dudX + I;
            auto C = transpose(F) * F;
            auto J = det(F);
            auto Cinv = inv(C);

            // Mooney-Rivlin: W = c1(I1-3) + c2(I2-3) + bulk_term
            const real_t I1 = tr(C);
            const real_t I2 = 0.5 * (I1 * I1 - tr(C * C));

            auto S = 2.0 * c1 * I + 2.0 * c2 * (I1 * I - C) + kappa * (J - 1) * J * Cinv;
            return F * S;
         }

         real_t kappa = 1e6; 
         real_t c1 = 0.8; // c1 = c1_ratio * mu
         real_t c2 = 0.2; // c2 = c2_ratio * mu
      };

      
      // Factory functions for easy material creation
      template <int dim>
      auto make_linear_elastic(real_t E, real_t nu = 0.3)
      {
         return LinearElastic<dim>{.nu = nu, .E = E};
      }

      template <int dim>
      auto make_saint_venant_kirchoff(real_t E, real_t nu = 0.3)
      {
         return SaintVenantKirchoff<dim>{.nu = nu, .E = E};
      }

      template <int dim>
      auto make_neo_hookean(real_t kappa, real_t c)
      {
         return NeoHookean<dim>{.kappa = kappa, .mu = 2.0 * c};
      }

      template <int dim>
      auto make_mooney_rivlin(real_t kappa, real_t c1, real_t c2)
      {
         return MooneyRivlin<dim>{.kappa = kappa, .c1 = c1, .c2 = c2};
      }



      // Helper function to get material name at compile time
      template <typename MaterialType>
      constexpr const char *get_material_name()
      {
         return MaterialType::name;
      }


      // Material variant for runtime polymorphism without virtual functions
      template <int dim>
      using MaterialVariant = std::variant<
         LinearElastic<dim>,
         SaintVenantKirchoff<dim>,
         NeoHookean<dim>,
         MooneyRivlin<dim>
      >;



   } // namespace elasticity_ecm2
} // namespace mfem

