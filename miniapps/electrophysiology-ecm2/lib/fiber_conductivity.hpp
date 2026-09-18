#pragma once
#include "mfem.hpp"
#include <cmath>

// References must outlive this coefficient. Works with serial or parallel GFs.
// sigma values include any scaling required by the caller's diffusion equation.
class FiberConductivity : public mfem::SymmetricMatrixCoefficient
{
   const mfem::GridFunction &fiber, &sheet;
   mfem::real_t sf, ss, sn;
public:
   FiberConductivity(const mfem::GridFunction &f, const mfem::GridFunction &s,
                     mfem::real_t longitudinal, mfem::real_t transverse,
                     mfem::real_t normal)
      : mfem::SymmetricMatrixCoefficient(3), fiber(f), sheet(s),
        sf(longitudinal), ss(transverse), sn(normal)
   {
      MFEM_VERIFY(f.VectorDim() == 3 && s.VectorDim() == 3, "Expected 3-vector fields");
      MFEM_VERIFY(f.FESpace()->GetMesh() == s.FESpace()->GetMesh(), "Fields must share a mesh");
      MFEM_VERIFY(sf > 0 && ss > 0 && sn > 0, "Conductivities must be positive");
   }
   using mfem::SymmetricMatrixCoefficient::Eval;
   void Eval(mfem::DenseSymmetricMatrix &K, mfem::ElementTransformation &T,
             const mfem::IntegrationPoint &ip) override
   {
      mfem::Vector f(3), s(3), n(3);
      fiber.GetVectorValue(T, ip, f);
      sheet.GetVectorValue(T, ip, s);
      const auto fn = f.Norml2();
      MFEM_VERIFY(std::isfinite(fn) && fn > 1e-12, "Invalid interpolated fiber");
      f /= fn;
      s.Add(-(f*s), f);
      const auto snorm = s.Norml2();
      MFEM_VERIFY(std::isfinite(snorm) && snorm > 1e-12, "Invalid interpolated sheet");
      s /= snorm;
      n[0] = f[1]*s[2]-f[2]*s[1];
      n[1] = f[2]*s[0]-f[0]*s[2];
      n[2] = f[0]*s[1]-f[1]*s[0];
      K.SetSize(3);
      for (int i=0; i<3; ++i)
         for (int j=i; j<3; ++j)
            K(i,j) = sf*f[i]*f[j] + ss*s[i]*s[j] + sn*n[i]*n[j];
   }
};
