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


    

}


}




