// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2019 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

#include <ibamr/config.h>

#include "ADS/IntegrateFunction.h"
#include "ADS/ls_functions.h"

#include "ibamr/app_namespaces.h"

#include "QFcn.h"

#include <SAMRAI_config.h>

#include <array>

namespace LS
{
/////////////////////////////// PUBLIC ///////////////////////////////////////

QFcn::QFcn(const string& object_name, Pointer<Database> input_db) : LSCartGridFunction(object_name)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(!d_object_name.empty());
#endif
    d_constant = input_db->getBool("use_constant");
    if (d_constant) d_num = input_db->getDouble("num");
    return;
} // QFcn

void
QFcn::setDataOnPatch(const int data_idx,
                     Pointer<hier::Variable<NDIM>> /*var*/,
                     Pointer<Patch<NDIM>> patch,
                     const double data_time,
                     const bool initial_time,
                     Pointer<PatchLevel<NDIM>> /*level*/)
{
    Pointer<CellData<NDIM, double>> Q_data = patch->getPatchData(data_idx);
    Q_data->fillAll(0.0);
    if (initial_time) return;
    auto fcn = [this](double y) -> double {
        return std::exp(10.0 * -y) / (std::exp(2.75 * 10.0) + std::exp(10.0 * -y));
    };

    const Box<NDIM>& patch_box = patch->getBox();
    const hier::Index<NDIM>& idx_low = patch->getBox().lower();
    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();

    const double* const xlow = pgeom->getXLower();
    const double* const dx = pgeom->getDx();

    Pointer<NodeData<NDIM, double>> ls_data = patch->getPatchData(d_ls_idx);
    Pointer<CellData<NDIM, double>> vol_data = patch->getPatchData(d_vol_idx);

    for (CellIterator<NDIM> ci(patch_box); ci; ci++)
    {
        const CellIndex<NDIM>& idx = ci();
        if ((*vol_data)(idx) > 0.0)
        {
            double y = xlow[1] + dx[1] * (static_cast<double>(idx(1) - idx_low(1)) + 0.5);
            if (d_constant)
                (*Q_data)(idx) = d_num;
            else
                (*Q_data)(idx) = fcn(y);
        }
    }

    return;
} // setDataOnPatch
} // namespace LS
