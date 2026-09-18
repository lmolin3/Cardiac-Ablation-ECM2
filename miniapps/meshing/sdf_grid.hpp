// A level set sampled on a uniform Cartesian grid, read from a file.
//
// The analytic level sets in mesh-fitting.hpp only cover shapes someone wrote a
// formula for. Real geometry -- a segmentation, or a voxelised mesh such as
// InSilicoHeartGen / MonoAlg3D produce -- arrives as samples on a grid instead.
// This reads such a field and evaluates it anywhere, so it can be handed to
// anything that takes a Coefficient.
//
// The file is written by mesh_to_sdf.py:
//
//   MFEM sampled level set v1
//   dims <nx> <ny> <nz>
//   origin <x0> <y0> <z0>
//   spacing <h>
//   data float64-le x-fastest
//   <nx*ny*nz doubles, little-endian, x fastest>
//
// Evaluation is trilinear inside the grid and linearly extrapolated outside it,
// deliberately: clamping instead would flatten the field beyond the box, and a
// node that wandered out would then feel no gradient and never come back.

#ifndef MFEM_SDF_GRID_HPP
#define MFEM_SDF_GRID_HPP

#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mfem
{

class SampledLevelSet
{
public:
   SampledLevelSet() : spacing(0.0)
   { for (int d = 0; d < 3; d++) { n[d] = 0; origin[d] = 0.0; } }

   void Load(const char *filename)
   {
      std::ifstream in(filename, std::ios::binary);
      MFEM_VERIFY(in.good(), "cannot open sampled level set file: " << filename);

      std::string line;
      std::getline(in, line);
      MFEM_VERIFY(line.rfind("MFEM sampled level set", 0) == 0,
                  filename << ": not a sampled level set file (got \"" << line << "\")");

      bool have_dims = false, have_origin = false, have_spacing = false;
      while (std::getline(in, line))
      {
         std::istringstream is(line);
         std::string key;
         is >> key;
         if (key == "dims")
         {
            is >> n[0] >> n[1] >> n[2];
            have_dims = true;
         }
         else if (key == "origin")
         {
            is >> origin[0] >> origin[1] >> origin[2];
            have_origin = true;
         }
         else if (key == "spacing")
         {
            is >> spacing;
            have_spacing = true;
         }
         else if (key == "data")
         {
            std::string dtype, order;
            is >> dtype >> order;
            MFEM_VERIFY(dtype == "float64-le" && order == "x-fastest",
                        filename << ": unsupported data layout \"" << dtype
                        << " " << order << "\"");
            break;
         }
      }
      MFEM_VERIFY(have_dims && have_origin && have_spacing,
                  filename << ": incomplete header");
      MFEM_VERIFY(n[0] > 1 && n[1] > 1 && n[2] > 1 && spacing > 0.0,
                  filename << ": bad grid " << n[0] << "x" << n[1] << "x" << n[2]
                  << " spacing " << spacing);

      const long np = (long)n[0] * n[1] * n[2];
      std::vector<double> buf(np);
      in.read(reinterpret_cast<char *>(buf.data()), np * sizeof(double));
      MFEM_VERIFY(in.gcount() == (std::streamsize)(np * sizeof(double)),
                  filename << ": expected " << np << " doubles, got "
                  << in.gcount() / sizeof(double));
      val.assign(buf.begin(), buf.end());
   }

   int Size() const { return (int)val.size(); }
   const int *Dims() const { return n; }
   real_t Spacing() const { return spacing; }
   void GetOrigin(Vector &o) const
   { o.SetSize(3); for (int d = 0; d < 3; d++) { o(d) = origin[d]; } }

   void GetBoundingBox(Vector &lo, Vector &hi) const
   {
      lo.SetSize(3); hi.SetSize(3);
      for (int d = 0; d < 3; d++)
      {
         lo(d) = origin[d];
         hi(d) = origin[d] + spacing * (n[d] - 1);
      }
   }

   real_t Min() const { real_t m = val[0]; for (auto v : val) { m = std::min(m, v); } return m; }
   real_t Max() const { real_t m = val[0]; for (auto v : val) { m = std::max(m, v); } return m; }

   /// Trilinear value at a physical point, linearly extrapolated outside.
   real_t Eval(const Vector &x) const
   {
      int c[3];
      real_t t[3];
      for (int d = 0; d < 3; d++)
      {
         const real_t s = (x(d) - origin[d]) / spacing;
         // Clamp the CELL, not the local coordinate: t leaves [0,1] outside the
         // grid and the interpolant becomes a linear extrapolation.
         int i = (int)std::floor(s);
         if (i < 0) { i = 0; }
         if (i > n[d] - 2) { i = n[d] - 2; }
         c[d] = i;
         t[d] = s - i;
      }
      const real_t w0[3] = {1.0 - t[0], 1.0 - t[1], 1.0 - t[2]};
      real_t f = 0.0;
      for (int k = 0; k < 2; k++)
      {
         for (int j = 0; j < 2; j++)
         {
            for (int i = 0; i < 2; i++)
            {
               const real_t w = (i ? t[0] : w0[0]) * (j ? t[1] : w0[1]) *
                                (k ? t[2] : w0[2]);
               f += w * At(c[0] + i, c[1] + j, c[2] + k);
            }
         }
      }
      return f;
   }

private:
   real_t At(int i, int j, int k) const
   {
      return val[(size_t)i + (size_t)n[0] * ((size_t)j + (size_t)n[1] * k)];
   }

   int n[3];
   real_t origin[3];
   real_t spacing;
   std::vector<real_t> val;
};

/// Coefficient wrapper. Held as a file-scope object where a plain
/// FunctionCoefficient callback is required.
class SampledLevelSetCoefficient : public Coefficient
{
public:
   SampledLevelSetCoefficient(const SampledLevelSet &g) : grid(g) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(3);
      T.Transform(ip, x);
      return grid.Eval(x);
   }

private:
   const SampledLevelSet &grid;
};

} // namespace mfem

#endif
