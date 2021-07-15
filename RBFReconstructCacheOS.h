#ifndef included_RBFReconstructCacheOS
#define included_RBFReconstructCacheOS

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "CCAD/ReconstructCache.h"
#include "CCAD/ls_functions.h"

#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/ibtk_utilities.h"

#include "CellData.h"
#include "CellIndex.h"
#include "IntVector.h"
#include "Patch.h"
#include "PatchHierarchy.h"
#include "tbox/Pointer.h"

#include <Eigen/Dense>

#include <map>
#include <string>
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Class RBFReconstructCacheOS caches the data necessary to form RBF
 * reconstructions of data.
 *
 * Only uses data that is on one side of the tangent plane
 */
class RBFReconstructCacheOS : public CCAD::ReconstructCache
{
public:
    RBFReconstructCacheOS() = default;

    RBFReconstructCacheOS(int ls_idx,
                          int vol_idx,
                          SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                          bool use_centroids = true);

    ~RBFReconstructCacheOS() = default;

    /*!
     * \brief Deleted copy constructor.
     */
    RBFReconstructCacheOS(const RBFReconstructCacheOS& from) = delete;

    void cacheData() override;
    void clearCache() override;

    double reconstructOnIndex(IBTK::VectorNd x_loc,
                              const SAMRAI::hier::Index<NDIM>& idx,
                              const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                              SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch) override;

    // DEBUGGING
    void setTime(double t)
    {
        d_time = t;
    }

private:
    void cacheData(IBTK::VectorNd x_loc,
                   const SAMRAI::hier::Index<NDIM>& idx,
                   const SAMRAI::pdat::CellData<NDIM, double>& Q_data,
                   SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch);

    std::vector<std::map<CCAD::IndexList, std::vector<SAMRAI::hier::Index<NDIM>>>> d_reconstruct_idxs_map_vec;

    // DEBUGGING
    double d_time = 0.0;
};
#endif
