#include "navierstokes_preconditioners.hpp"

using namespace mfem;

/// PCD PRECONDITIONER
// Implementation of PCD constructor
PCD::PCD(Solver &Mp_inv, Solver &Lp_inv, OperatorHandle &Fp) : NavierStokesPC(Mp_inv.Width()),
                                                               Mp_inv(Mp_inv),
                                                               Lp_inv(Lp_inv),
                                                               Fp(Fp) {}

// Implementation of PCD::Mult
void PCD::Mult(const Vector &x, Vector &y) const
{
   z.SetSize(y.Size());
   w.SetSize(y.Size());

   Lp_inv.Mult(x, z);
   Fp->Mult(z, w);
   Mp_inv.Mult(w, y);
}

// Implementation of PCDBuilder constructor
PCDBuilder::PCDBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs,
                       Coefficient *mass_coeff, Coefficient *diff_coeff) : PCBuilder(),
                                                                           mp_form(&pres_fes),
                                                                           lp_form(&pres_fes),
                                                                           fp_form(&pres_fes)
{
   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.Finalize();
   mp_form.FormSystemMatrix(pres_ess_tdofs, Mp);

   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.Finalize();
   lp_form.FormSystemMatrix(pres_ess_tdofs, Lp);

   fp_form.AddDomainIntegrator(new MassIntegrator(*mass_coeff));
   fp_form.AddDomainIntegrator(new DiffusionIntegrator(*diff_coeff));
   fp_form.Assemble();
   fp_form.Finalize();
   fp_form.FormSystemMatrix(pres_ess_tdofs, Fp);

   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);

   pcd = new PCD(*Mp_inv, *Lp_inv, Fp);
}

// Implementation of PCDBuilder constructor with default coefficients
PCDBuilder::PCDBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs) : PCBuilder(),
                                                                                     mp_form(&pres_fes),
                                                                                     lp_form(&pres_fes),
                                                                                     fp_form(&pres_fes)
{
   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.Finalize();
   mp_form.FormSystemMatrix(pres_ess_tdofs, Mp);

   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.Finalize();
   lp_form.FormSystemMatrix(pres_ess_tdofs, Lp);

   fp_form.AddDomainIntegrator(new MassIntegrator);
   fp_form.AddDomainIntegrator(new DiffusionIntegrator);
   fp_form.Assemble();
   fp_form.Finalize();
   fp_form.FormSystemMatrix(pres_ess_tdofs, Fp);

   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);

   pcd = new PCD(*Mp_inv, *Lp_inv, Fp);
}

// Implementation of PCDBuilder::GetSolver
NavierStokesPC &PCDBuilder::GetSolver() { return *pcd; }

// Implementation of PCDBuilder destructor
PCDBuilder::~PCDBuilder()
{
   delete pcd;
   delete Lp_inv;
   delete Mp_inv;
}

/// CAHOUET CHABARD PRECONDITIONER
// Implementation of CahouetChabardPC constructor
CahouetChabardPC::CahouetChabardPC(Solver &Mp_inv, Solver &Lp_inv,
                                   Array<int> pres_ess_tdofs, double dt) : NavierStokesPC(Mp_inv.Height()),
                                                                           Mp_inv(Mp_inv),
                                                                           Lp_inv(Lp_inv),
                                                                           z(Mp_inv.Height()),
                                                                           dt(dt),
                                                                           pres_ess_tdofs(pres_ess_tdofs) {}

// Implementation of CahouetChabardPC::Mult
void CahouetChabardPC::Mult(const Vector &x, Vector &y) const
{
   z.SetSize(y.Size());

   Lp_inv.Mult(x, y);
   y /= dt;
   Mp_inv.Mult(x, z);
   y += z;

   for (int i = 0; i < pres_ess_tdofs.Size(); i++)
   {
      y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
   }
}

// Implementation of CahouetChabardBuilder constructor
CahouetChabardBuilder::CahouetChabardBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs, double dt) : mp_form(&pres_fes),
                                                                                                                      lp_form(&pres_fes)
{
   Array<int> empty;

   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.Finalize();
   mp_form.FormSystemMatrix(empty, Mp);

   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.Finalize();
   lp_form.FormSystemMatrix(empty, Lp);

   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);

   cahouet_chabard = new CahouetChabardPC(*Mp_inv, *Lp_inv, pres_ess_tdofs, dt);
}

// Implementation of CahouetChabardBuilder::GetSolver
NavierStokesPC &CahouetChabardBuilder::GetSolver() { return *cahouet_chabard; }

// Implementation of CahouetChabardBuilder destructor
CahouetChabardBuilder::~CahouetChabardBuilder()
{
   delete cahouet_chabard;
   delete Lp_inv;
   delete Mp_inv;
}

/// APPROXIMATE INVERSE PRECONDITIONER
// Implementation of SchurApproxInvPC constructor
SchurApproxInvPC::SchurApproxInvPC(Solver &Mp_inv, Solver &Lp_inv,
                                   Array<int> pres_ess_tdofs, double dt) : NavierStokesPC(Mp_inv.Height()),
                                                                           Mp_inv(Mp_inv),
                                                                           Lp_inv(Lp_inv),
                                                                           z(Mp_inv.Height()),
                                                                           dt(dt),
                                                                           pres_ess_tdofs(pres_ess_tdofs) {}

// Implementation of SchurApproxInvPC::Mult
void SchurApproxInvPC::Mult(const Vector &x, Vector &y) const
{
   z.SetSize(y.Size());

   Mp_inv.Mult(x, z);
   z *= dt;
   Lp_inv.Mult(x, y);
   y += z;

   for (int i = 0; i < pres_ess_tdofs.Size(); i++)
   {
      y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
   }
}

// Implementation of SchurApproxInvBuilder constructor
SchurApproxInvBuilder::SchurApproxInvBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs, double dt) : mp_form(&pres_fes),
                                                                                                                      lp_form(&pres_fes)
{
   Array<int> empty;

   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.Finalize();
   mp_form.FormSystemMatrix(empty, Mp);

   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.Finalize();
   lp_form.FormSystemMatrix(empty, Lp);

   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);

   schur_approx_inv = new SchurApproxInvPC(*Mp_inv, *Lp_inv, pres_ess_tdofs, dt);
}

// Implementation of SchurApproxInvBuilder::GetSolver
NavierStokesPC &SchurApproxInvBuilder::GetSolver() { return *schur_approx_inv; }

// Implementation of SchurApproxInvBuilder destructor
SchurApproxInvBuilder::~SchurApproxInvBuilder()
{
   delete schur_approx_inv;
   delete Lp_inv;
   delete Mp_inv;
}

/// PRESSURE MASS PRECONDITIONER
// Implementation of PMassPC constructor
PMassPC::PMassPC(Solver &Mp_inv) : NavierStokesPC(Mp_inv.Width()),
                                   Mp_inv(Mp_inv) {}

// Implementation of PMassPC::Mult
void PMassPC::Mult(const Vector &x, Vector &y) const
{
   Mp_inv.Mult(x, y);
}

// Implementation of PMassBuilder constructor
PMassBuilder::PMassBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs) : mp_form(&pres_fes)
{
   mp_form.AddDomainIntegrator(new MassIntegrator);
   mp_form.Assemble();
   mp_form.Finalize();
   mp_form.FormSystemMatrix(pres_ess_tdofs, Mp);
#ifndef MFEM_USE_SUITESPARSE
   Mp_inv = new HypreBoomerAMG(*Mp.As<HypreParMatrix>());
#else
   Mp.As<HypreParMatrix>()->GetDiag(M_local);
   Mp_inv = new UMFPackSolver(M_local);
   dynamic_cast<HypreBoomerAMG *>(Mp_inv)->SetPrintLevel(0);
#endif
   pmass = new PMassPC(*Mp_inv);
}

// Implementation of PMassBuilder::GetSolver
NavierStokesPC &PMassBuilder::GetSolver() { return *pmass; }

// Implementation of PMassBuilder destructor
PMassBuilder::~PMassBuilder()
{
   delete pmass;
   delete Mp_inv;
}

/// PRESSURE LAPLACIAN PRECONDITIONER
// Implementation of PMassPC constructor
PLapPC::PLapPC(Solver &Lp_inv, double dt) : NavierStokesPC(Lp_inv.Width()),
                                            Lp_inv(Lp_inv), dt(dt) {}

// Implementation of PMassPC::Mult
void PLapPC::Mult(const Vector &x, Vector &y) const
{
   Lp_inv.Mult(x, y);
   y /= dt;
}

// Implementation of PMassBuilder constructor
PLapBuilder::PLapBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs,
                         double dt) : lp_form(&pres_fes)
{
   lp_form.AddDomainIntegrator(new DiffusionIntegrator);
   lp_form.Assemble();
   lp_form.Finalize();
   lp_form.FormSystemMatrix(pres_ess_tdofs, Lp);
#ifndef MFEM_USE_SUITESPARSE
   Lp_inv = new HypreBoomerAMG(*Lp.As<HypreParMatrix>());
#else
   Lp.As<HypreParMatrix>()->GetDiag(L_local);
   Lp_inv = new UMFPackSolver(L_local);
   dynamic_cast<HypreBoomerAMG *>(Lp_inv)->SetPrintLevel(0);
#endif
   plap = new PLapPC(*Lp_inv, dt);
}

// Implementation of PLapBuilder::GetSolver
NavierStokesPC &PLapBuilder::GetSolver() { return *plap; }

// Implementation of PLapBuilder destructor
PLapBuilder::~PLapBuilder()
{
   delete plap;
   delete Lp_inv;
}