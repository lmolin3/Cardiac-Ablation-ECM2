//                  ------------------------------------------
//                  Sampled level set -> MFEM GridFunction
//                  ------------------------------------------
//
// Takes the Cartesian-grid level set written by mesh_to_sdf.py and turns it
// into an MFEM GridFunction on a background hex mesh, plus a ParaView
// collection to look at it in.
//
// Why a background mesh rather than the level set on the mesh being fitted: the
// level set has to stay put in space while the mesh moves through it, and its
// resolution should not be tied to the resolution of the mesh being repaired.
// TMOP surface fitting reads phi, grad phi and Hess phi, so the background
// space wants order >= 2 -- the Hessian of a trilinear field is junk.
//
// Sample runs:
//   sdf-to-gf -sdf biv_2.0.sdf -o biv_2.0_ls -hb 2.0 -lo 2 -pv
//   sdf-to-gf -sdf biv_2.0.sdf -o biv_2.0_ls -m biv_2.0.mesh -pv

#include "sdf_grid.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   const char *sdf_file  = "";
   const char *out_pfx   = "level_set";
   const char *src_mesh  = "";
   real_t hb             = -1.0;
   int ls_order          = 2;
   bool paraview         = false;
   const char *devopt    = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&sdf_file, "-sdf", "--sdf-file",
                  "Sampled level set file from mesh_to_sdf.py.");
   args.AddOption(&out_pfx, "-o", "--output-prefix",
                  "Prefix for <pfx>.mesh, <pfx>.gf and the ParaView directory.");
   args.AddOption(&src_mesh, "-m", "--source-mesh",
                  "Optional staircase mesh the level set came from; exported "
                  "alongside so the two can be overlaid.");
   args.AddOption(&hb, "-hb", "--bg-element-size",
                  "Background mesh element size (default: 2 x grid spacing).");
   args.AddOption(&ls_order, "-lo", "--level-set-order",
                  "Polynomial order of the level set space (default 2).");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Write a ParaView collection.");
   args.AddOption(&devopt, "-d", "--device", "Device configuration string.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   Device device(devopt);
   device.Print();

   MFEM_VERIFY(strlen(sdf_file) > 0, "-sdf is required");

   SampledLevelSet grid;
   grid.Load(sdf_file);
   Vector lo, hi;
   grid.GetBoundingBox(lo, hi);
   const int *n = grid.Dims();
   cout << "\nsampled level set " << sdf_file << "\n"
        << "  grid    " << n[0] << " x " << n[1] << " x " << n[2]
        << " = " << grid.Size() << " samples at h = " << grid.Spacing() << "\n"
        << "  box     [" << lo(0) << ", " << hi(0) << "] x ["
        << lo(1) << ", " << hi(1) << "] x [" << lo(2) << ", " << hi(2) << "]\n"
        << "  phi     [" << grid.Min() << ", " << grid.Max() << "]" << endl;

   if (hb <= 0.0) { hb = 2.0 * grid.Spacing(); }

   // Background mesh: a Cartesian hex grid over exactly the sampled box, so the
   // level set is never extrapolated inside it.
   int ne[3];
   for (int d = 0; d < 3; d++)
   {
      ne[d] = max(1, (int)std::round((hi(d) - lo(d)) / hb));
   }
   Mesh bg = Mesh::MakeCartesian3D(ne[0], ne[1], ne[2], Element::HEXAHEDRON,
                                   hi(0) - lo(0), hi(1) - lo(1), hi(2) - lo(2));
   // MakeCartesian3D starts at the origin; shift it onto the sampled box.
   // Done on the vertices, before any nodal GridFunction exists, so there is no
   // byNODES/byVDIM ordering to get wrong.
   for (int i = 0; i < bg.GetNV(); i++)
   {
      real_t *v = bg.GetVertex(i);
      for (int d = 0; d < 3; d++) { v[d] += lo(d); }
   }

   H1_FECollection fec(ls_order, 3);
   FiniteElementSpace fes(&bg, &fec);
   GridFunction phi(&fes);
   SampledLevelSetCoefficient ls_coeff(grid);
   phi.ProjectCoefficient(ls_coeff);

   cout << "\nbackground mesh " << ne[0] << " x " << ne[1] << " x " << ne[2]
        << " = " << bg.GetNE() << " hexes at h = "
        << (hi(0) - lo(0)) / ne[0] << "\n"
        << "  level set   H1 order " << ls_order << ", " << fes.GetNDofs()
        << " dofs\n"
        << "  node spacing " << (hi(0) - lo(0)) / (ne[0] * ls_order)
        << " vs grid spacing " << grid.Spacing() << "\n"
        << "  phi on the gf [" << phi.Min() << ", " << phi.Max() << "]" << endl;

   // TMOP also builds a gradient and a Hessian from this field. Their storage
   // is what actually decides whether the background mesh is affordable.
   const real_t mb = fes.GetNDofs() * sizeof(real_t) / 1e6;
   cout << "  memory      " << fixed << setprecision(1) << mb << " MB for phi, "
        << 13 * mb << " MB with grad + Hessian" << endl;

   // How much of the sampled field survived the projection. Both the samples
   // and the projected field are evaluated on a much finer space on the same
   // mesh, so this needs no point searching. The number that matters is the one
   // in the band around the zero isosurface: that is the only part of the field
   // TMOP ever reads.
   {
      H1_FECollection fec_f(ls_order + 2, 3);
      FiniteElementSpace fes_f(&bg, &fec_f);
      GridFunction exact_f(&fes_f), phi_f(&fes_f);
      exact_f.ProjectCoefficient(ls_coeff);
      GridFunctionCoefficient phi_c(&phi);
      phi_f.ProjectCoefficient(phi_c);

      const real_t bands[3] = {0.5 * hb, 2.0 * hb, infinity()};
      real_t e_max[3] = {0.0, 0.0, 0.0}, e_sum[3] = {0.0, 0.0, 0.0};
      long nb[3] = {0, 0, 0};
      for (int i = 0; i < fes_f.GetNDofs(); i++)
      {
         const real_t e = fabs(phi_f(i) - exact_f(i));
         for (int b = 0; b < 3; b++)
         {
            if (fabs(exact_f(i)) < bands[b])
            {
               e_max[b] = max(e_max[b], e);
               e_sum[b] += e;
               nb[b]++;
            }
         }
      }
      cout << "  projection error vs the samples:" << endl;
      for (int b = 0; b < 3; b++)
      {
         cout << "    ";
         if (b == 2) { cout << "whole box     "; }
         else
         {
            ostringstream os;
            os << "|phi| < " << fixed << setprecision(2) << bands[b] << " mm";
            cout << left << setw(14) << os.str() << right;
         }
         cout << scientific << setprecision(3)
              << "  avg " << (nb[b] ? e_sum[b] / nb[b] : 0.0)
              << ", max " << e_max[b] << " mm   (" << nb[b] << " points)"
              << endl;
      }
   }

   {
      ofstream mout(string(out_pfx) + ".mesh");
      mout.precision(16);
      bg.Print(mout);
      ofstream gout(string(out_pfx) + ".gf");
      gout.precision(16);
      phi.Save(gout);
      cout << "\nwrote " << out_pfx << ".mesh and " << out_pfx << ".gf" << endl;
   }

   if (paraview)
   {
      ParaViewDataCollection pv(string(out_pfx) + "_bg", &bg);
      pv.SetPrefixPath("ParaView");
      pv.SetLevelsOfDetail(max(1, ls_order));
      pv.SetDataFormat(VTKFormat::BINARY);
      pv.SetHighOrderOutput(ls_order > 1);
      pv.RegisterField("level_set", &phi);
      pv.SetCycle(0);
      pv.SetTime(0.0);
      pv.Save();
      cout << "wrote ParaView/" << out_pfx << "_bg -- Contour 'level_set' at 0"
           << endl;

      if (strlen(src_mesh) > 0)
      {
         Mesh sm(src_mesh, 1, 1, false);
         L2_FECollection l2(0, 3);
         FiniteElementSpace l2fes(&sm, &l2);
         GridFunction attr(&l2fes);
         for (int i = 0; i < sm.GetNE(); i++) { attr(i) = sm.GetAttribute(i); }
         ParaViewDataCollection pvs(string(out_pfx) + "_source", &sm);
         pvs.SetPrefixPath("ParaView");
         pvs.SetDataFormat(VTKFormat::BINARY);
         pvs.RegisterField("attribute", &attr);
         pvs.SetCycle(0);
         pvs.SetTime(0.0);
         pvs.Save();
         cout << "wrote ParaView/" << out_pfx << "_source -- the staircase mesh"
              << endl;
      }
   }

   return 0;
}
