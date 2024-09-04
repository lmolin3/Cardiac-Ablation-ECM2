#include "utils.hpp"

namespace mfem
{

namespace ecm2_utils
{
    // Mult and AddMult for full matrix (using matrices modified with WliminateRowsCols)
    void FullMult(HypreParMatrix* mat, HypreParMatrix* mat_e, Vector &x, Vector &y)
    {
        mat->Mult(x, y);        // y =  mat x
        mat_e->AddMult(x, y);   // y += mat_e x
    }


    void FullAddMult(HypreParMatrix* mat, HypreParMatrix* mat_e, Vector &x, Vector &y, double a)
    {
        mat->AddMult(x, y, a);     // y +=  a mat x
        mat_e->AddMult(x, y, a);   // y += a mat_e x
    }


    // Class containing computation of potential quantities of interest
    QuantitiesOfInterest::QuantitiesOfInterest(ParMesh *pmesh)
    {
        H1_FECollection h1fec(1);
        ParFiniteElementSpace h1fes(pmesh, &h1fec);

        onecoeff.constant = 1.0;
        mass_lf = new ParLinearForm(&h1fes);
        mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
        mass_lf->Assemble();

        ParGridFunction one_gf(&h1fes);
        one_gf.ProjectCoefficient(onecoeff);

        volume = mass_lf->operator()(one_gf);
    };

    double QuantitiesOfInterest::ComputeKineticEnergy(ParGridFunction &v)
    {
        Vector velx, vely, velz;
        double integ = 0.0;
        const FiniteElement *fe;
        ElementTransformation *T;
        FiniteElementSpace *fes = v.FESpace();

        for (int i = 0; i < fes->GetNE(); i++)
        {
            fe = fes->GetFE(i);
            int intorder = 2 * fe->GetOrder();
            const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

            v.GetValues(i, *ir, velx, 1);
            v.GetValues(i, *ir, vely, 2);
            v.GetValues(i, *ir, velz, 3);

            T = fes->GetElementTransformation(i);
            for (int j = 0; j < ir->GetNPoints(); j++)
            {
                const IntegrationPoint &ip = ir->IntPoint(j);
                T->SetIntPoint(&ip);

                double vel2 = velx(j) * velx(j) + vely(j) * vely(j)
                            + velz(j) * velz(j);

                integ += ip.weight * T->Weight() * vel2;
            }
        }

        double global_integral = 0.0;
        MPI_Allreduce(&integ,
                        &global_integral,
                        1,
                        MPI_DOUBLE,
                        MPI_SUM,
                        MPI_COMM_WORLD);

        return 0.5 * global_integral / volume;
    };


    double QuantitiesOfInterest::ComputeCFL(ParGridFunction &u, double dt)
    {
    ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
    FiniteElementSpace *fes = u.FESpace();
    int vdim = fes->GetVDim();

    Vector ux, uy, uz;
    Vector ur, us, ut;
    double cflx = 0.0;
    double cfly = 0.0;
    double cflz = 0.0;
    double cflm = 0.0;
    double cflmax = 0.0;

    for (int e = 0; e < fes->GetNE(); ++e)
    {
        const FiniteElement *fe = fes->GetFE(e);
        const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                fe->GetOrder());
        ElementTransformation *tr = fes->GetElementTransformation(e);

        u.GetValues(e, ir, ux, 1);
        ur.SetSize(ux.Size());
        u.GetValues(e, ir, uy, 2);
        us.SetSize(uy.Size());
        if (vdim == 3)
        {
            u.GetValues(e, ir, uz, 3);
            ut.SetSize(uz.Size());
        }

        double hmin = pmesh_u->GetElementSize(e, 1) /
                        (double) fes->GetElementOrder(0);

        for (int i = 0; i < ir.GetNPoints(); ++i)
        {
            const IntegrationPoint &ip = ir.IntPoint(i);
            tr->SetIntPoint(&ip);
            const DenseMatrix &invJ = tr->InverseJacobian();
            const double detJinv = 1.0 / tr->Jacobian().Det();

            if (vdim == 2)
            {
                ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
                us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
            }
            else if (vdim == 3)
            {
                ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)
                        + uz(i) * invJ(2, 0))
                        * detJinv;
                us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)
                        + uz(i) * invJ(2, 1))
                        * detJinv;
                ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2)
                        + uz(i) * invJ(2, 2))
                        * detJinv;
            }

            cflx = fabs(dt * ux(i) / hmin);
            cfly = fabs(dt * uy(i) / hmin);
            if (vdim == 3)
            {
                cflz = fabs(dt * uz(i) / hmin);
            }
            cflm = cflx + cfly + cflz;
            cflmax = fmax(cflmax, cflm);
        }
    }

    double cflmax_global = 0.0;
    MPI_Allreduce(&cflmax,
                    &cflmax_global,
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    pmesh_u->GetComm());

    return cflmax_global;
    }

    };

}







