// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2020 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_CCAD_RBFOneSidedReconstructions
#define included_CCAD_RBFOneSidedReconstructions

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include "CCAD/AdvectiveReconstructionOperator.h"
#include "CCAD/ls_utilities.h"
#include "CCAD/reconstructions.h"

#include "CellVariable.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Class RBFOneSidedReconstructions is a abstract class for an
 * implementation of a convective differencing operator.
 */
class RBFOneSidedReconstructions : public CCAD::AdvectiveReconstructionOperator
{
public:
    /*!
     * \brief Class constructor.
     */
    RBFOneSidedReconstructions(std::string object_name, Reconstruct::RBFPolyOrder rbf_poly_order, int stencil_size);

    /*!
     * \brief Destructor.
     */
    ~RBFOneSidedReconstructions();

    /*!
     * \brief Deletec Operators
     */
    //\{
    RBFOneSidedReconstructions() = delete;
    RBFOneSidedReconstructions(const RBFOneSidedReconstructions& from) = delete;
    RBFOneSidedReconstructions& operator=(const RBFOneSidedReconstructions& that) = delete;
    //\}

    /*!
     * \brief Initialize operator.
     */
    void allocateOperatorState(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                               double current_time,
                               double new_time) override;

    /*!
     * \brief Deinitialize operator
     */
    void deallocateOperatorState() override;

    /*!
     * \brief Compute N = u * grad Q.
     */
    void applyReconstruction(int Q_idx, int N_idx, int path_idx) override;

private:
    double oneSidedRBFReconstruct(IBTK::VectorNd x_loc,
                                  const SAMRAI::hier::Index<NDIM>& idx,
                                  SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                                  const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                                  const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                                  const SAMRAI::pdat::CellData<NDIM, double>& vol_data,
                                  const SAMRAI::pdat::CellData<NDIM, double>& vol_new_data,
                                  const SAMRAI::pdat::NodeData<NDIM, double>& ls_new_data);

    bool useIdx(const SAMRAI::hier::Index<NDIM>& idx,
                const SAMRAI::pdat::NodeData<NDIM, double>& ls_data,
                const SAMRAI::pdat::CellData<NDIM, double>& vol_data,
                const IBTK::VectorNd& normal_vec,
                const IBTK::VectorNd& midpt,
                double sgn = 0.0);

    bool appendIndexToList(std::vector<SAMRAI::pdat::CellIndex<NDIM>>& new_idxs,
                           const SAMRAI::pdat::CellIndex<NDIM>& idx,
                           const SAMRAI::pdat::CellIndex<NDIM>& orig_idx,
                           const SAMRAI::pdat::CellIndex<NDIM>& base_idx,
                           const SAMRAI::pdat::CellData<NDIM, double>& vol_data);

    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;
    Reconstruct::RBFPolyOrder d_rbf_order = Reconstruct::RBFPolyOrder::LINEAR;
    unsigned int d_rbf_stencil_size = 5;

    // Scratch data
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_Q_scr_var;
    int d_Q_scr_idx = IBTK::invalid_index;
};

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_CCAD_RBFOneSidedReconstructions
