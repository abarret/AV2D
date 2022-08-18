// Filename: VelocityBcCoefs.C
// Created on 04 May 2007 by Boyce Griffith

#include <ibamr/config.h>

#include <ibamr/app_namespaces.h>

#include "VelocityBcCoefs.h"

#include <CartesianPatchGeometry.h>
#include <SAMRAI_config.h>

/////////////////////////////// NAMESPACE ////////////////////////////////////

/////////////////////////////// STATIC ///////////////////////////////////////

/////////////////////////////// PUBLIC ///////////////////////////////////////

VelocityBcCoefs::VelocityBcCoefs(const CirculationModel* circ_model, const int comp_idx)
    : d_circ_model(circ_model), d_comp_idx(comp_idx)
{
    // intentionally blank
    return;
} // VelocityBcCoefs

VelocityBcCoefs::~VelocityBcCoefs()
{
    // intentionally blank
    return;
} // ~VelocityBcCoefs

void
VelocityBcCoefs::setBcCoefs(Pointer<ArrayData<NDIM, double>>& acoef_data,
                            Pointer<ArrayData<NDIM, double>>& bcoef_data,
                            Pointer<ArrayData<NDIM, double>>& gcoef_data,
                            const Pointer<Variable<NDIM>>& /*variable*/,
                            const Patch<NDIM>& patch,
                            const BoundaryBox<NDIM>& bdry_box,
                            double fill_time) const
{
    const int location_index = bdry_box.getLocationIndex();
    const int axis = location_index / 2;
    const int side = location_index % 2;
#if !defined(NDEBUG)
    TBOX_ASSERT(!acoef_data.isNull());
#endif
    const Box<NDIM>& bc_coef_box = acoef_data->getBox();
#if !defined(NDEBUG)
    TBOX_ASSERT(bcoef_data.isNull() || bc_coef_box == bcoef_data->getBox());
    TBOX_ASSERT(gcoef_data.isNull() || bc_coef_box == gcoef_data->getBox());
#endif
    const Box<NDIM>& patch_box = patch.getBox();
    const hier::Index<NDIM>& patch_lower = patch_box.lower();
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch.getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const x_lower = pgeom->getXLower();
    for (Box<NDIM>::Iterator bc(bc_coef_box); bc; bc++)
    {
        const hier::Index<NDIM>& i = bc();
        double dummy;
        double& a = (!acoef_data.isNull() ? (*acoef_data)(i, 0) : dummy);
        double& b = (!bcoef_data.isNull() ? (*bcoef_data)(i, 0) : dummy);
        double& g = (!gcoef_data.isNull() ? (*gcoef_data)(i, 0) : dummy);
        if (axis != 1)
        {
            a = 1.0;
            b = 0.0;
            g = 0.0;
        }
        else if (d_comp_idx != axis)
        {
            a = 1.0;
            b = 0.0;
            g = 0.0;
        }
        else if (d_comp_idx == axis)
        {
            const double psrc = (side == 0 ? 140965.0 : 135140.0); // d_circ_model->d_psrc[side];
            const double rsrc = d_circ_model->d_rsrc[side];
            const Point& posn = d_circ_model->d_posn[side];
            double X[NDIM];
            double r_sq = 0.0;
            for (int d = 0; d < NDIM; ++d)
            {
                X[d] = x_lower[d] + dx[d] * (double(i(d) - patch_lower(d)) + (d == axis ? 0.0 : 0.5));
                if (d != axis) r_sq += pow(X[d] - posn[d], 2.0);
            }
            const double r = sqrt(r_sq);
            a = (r <= rsrc) ? 0.0 : 1.0;
            b = (r <= rsrc) ? 1.0 : 0.0;
            g = (r <= rsrc) ? -psrc : 0.0;
        }
    }
    return;
} // setBcCoefs

IntVector<NDIM>
VelocityBcCoefs::numberOfExtensionsFillable() const
{
    return 128;
} // numberOfExtensionsFillable

/////////////////////////////// PROTECTED ////////////////////////////////////

/////////////////////////////// PRIVATE //////////////////////////////////////

/////////////////////////////// NAMESPACE ////////////////////////////////////

/////////////////////////////// TEMPLATE INSTANTIATION ///////////////////////

//////////////////////////////////////////////////////////////////////////////
