#ifndef included_CBFinder
#define included_CBFinder
#include "ibamr/config.h"

#include "ADS/SBSurfaceFluidCouplingManager.h"

#include "ibtk/FEDataManager.h"

#include "libmesh/boundary_mesh.h"
#include "libmesh/equation_systems.h"
#include "libmesh/explicit_system.h"
#include "libmesh/mesh_base.h"

namespace IBAMR
{
class CBFinder
{
public:
    CBFinder(std::string sf_name,
             libMesh::MeshBase* vol_mesh,
             std::shared_ptr<ADS::SBSurfaceFluidCouplingManager> sb_data_manager,
             IBTK::FEDataManager* fe_data_manager);

    CBFinder() = default;

    void setFEDataManager(IBTK::FEDataManager* fe_data_manager);

    void registerSFName(std::string sf_name);

    void registerSBData(libMesh::MeshBase* vol_mesh,
                        std::shared_ptr<ADS::SBSurfaceFluidCouplingManager> sb_data_manager);

    double findCB(const libMesh::TensorValue<double>& FF,
                  const libMesh::Point& X, // current location
                  const libMesh::Point& s, // reference location
                  libMesh::Elem* const elem,
                  double time);

    double maxCB() const;
    void printStatistics() const;

    void formHarmonicInterpolant();

private:
    void
    reportPETScKSPConvergedReason(const std::string& object_name, const KSPConvergedReason& reason, std::ostream& os);
    std::string d_sf_name;
    std::shared_ptr<ADS::SBSurfaceFluidCouplingManager> d_sb_data_manager;
    std::map<libMesh::dof_id_type, libMesh::dof_id_type> d_node_id_map;
    IBTK::FEDataManager* d_fe_data_manager = nullptr;
    libMesh::FEType d_elem_type;
    std::string d_phi_sys_name;
    double d_time = std::numeric_limits<double>::quiet_NaN();

    double d_max_CB = std::numeric_limits<double>::quiet_NaN();
    double d_min_CB = std::numeric_limits<double>::quiet_NaN();
    double d_avg_CB = 0.0;
    int d_tot_CB = std::numeric_limits<int>::max();
};
} // namespace IBAMR
#endif
