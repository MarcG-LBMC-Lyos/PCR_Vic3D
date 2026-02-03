from RigidRegistration import RigidRegistration
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyvista
import xlsxwriter


def outer_shell_tet2_tet1(elem_tab, node_coords):
    """
    Get outer shell of 1st or 2nd order tetrahedron mesh.
    :param elem_tab: Connection table of tetrahedron mesh.
    :param node_coords: Coordinates table of tetrahedron mesh
    :return: Connection table of triangle mesh of the outer shell of the original mesh.
    """
    #   Get Surface nodes
    elem_tab_tet1_node_arg = elem_tab[:, 1:5] - 1  # Node arg in element table of tetrahedron1 vertices
    cells = np.transpose((4 * np.ones(len(elem_tab)), *np.transpose(elem_tab_tet1_node_arg)))
    celltypes = 10 * np.ones(len(cells))  # Tet1
    points = np.array(node_coords)[:, 1:]
    cells_ravel = cells.ravel().astype(int)
    surf = pyvista.UnstructuredGrid(cells_ravel, celltypes, points)
    surface_node_ids = surf.extract_surface().point_data["vtkOriginalPointIds"]  # Get surface nodes
    surface_node_coords = node_coords[surface_node_ids]

    #   Get surface triangles
    elem_tab_tet1 = elem_tab[:, :5]  # Remove intermediate nodes of Tet2 to get Tet1 only
    init_shape_elem_tab = np.shape(elem_tab_tet1[:, 1:])
    mask_elem_surf = np.reshape(np.in1d(elem_tab_tet1[:, 1:].ravel(), surface_node_coords[:, 0]),
                                init_shape_elem_tab)  # Mask of nodes at surface in elem_tab
    mask_elem_surf_nodes = np.copy(mask_elem_surf)
    mask_elem_surf = np.sum(mask_elem_surf, axis=1) == 3  # Mask of tet elem which have a face (triangle) on the surface
    mask_elem_tab_surf = np.transpose((np.ones(len(mask_elem_surf_nodes)).astype(bool),
                                       *np.transpose(mask_elem_surf_nodes)))  # Full mask for elem_tab
    elem_tab_surf_triangle = np.array([elem_tab_tet1[i, :5][mask_elem_tab_surf[i]] for i in range(len(elem_tab_tet1))], dtype=object)[
        mask_elem_surf]  # elem_tab of surface triangle
    elem_tab_surf_triangle = np.array([elem_tab_surf_triangle[i] for i in range(len(elem_tab_surf_triangle))])

    return elem_tab_surf_triangle

def elem_table_ids2args(elem_table):
    node_ids = np.unique(np.ravel(elem_table[:, 1:]))
    new_node_ids = np.array(range(len(node_ids)))
    old2new_node_ids = {node_ids[i]: new_node_ids[i] for i in range(len(node_ids))}
    new2old_node_ids = {new_node_ids[i]: node_ids[i] for i in range(len(node_ids))}
    elem_table_as_args = np.array([[row[j] if j == 0 else old2new_node_ids[row[j]] for j in range(len(row))] for row in elem_table])

    return elem_table_as_args, old2new_node_ids, new2old_node_ids


def pcr_vic3d(vic3d_node_coords_, ansys_node_coords_, vic3d_node_deformations_, ansys_deformations_, ansys_elem_tab_,
              zyx_rot_eul_angles_init=(0, 0, 0), radius_smooth=0.0005, show_plot=True, save_img_path=False,
              save_deformation_path=None):
    """
    Match 2 surfaces (vic3d surface and ansys mesh surface) to compare their local deformation.
    :param vic3d_node_coords_: List of [node, coord1, coord2, coord3] from vic3d surface.
    :param ansys_node_coords_: List of [node, coord1, coord2, coord3] from ansys mesh surface.
    :param vic3d_node_deformations_: List of [node, deformation] from vic3d surface.
    :param ansys_deformations_: List of [elem, deformation] from ansys mesh surface.
    :param ansys_elem_tab_: List of [elem, node1, node2, ...] from ansys mesh surface.
    :param zyx_rot_eul_angles_init: Sequence of angle (z angle, y angle, x angle) for initial orientation of the ansys surface.
    :param radius_smooth: Radius of the sphere used to get vic3d equivalent deformation (must be length of an FEA elem).
    :return: A new mesh with values of the difference od deformations on similar nodes.
    """
    vic3d_node_coords = np.array(vic3d_node_coords_)
    ansys_node_coords = np.array(ansys_node_coords_)
    vic3d_node_deformations = np.array(vic3d_node_deformations_)
    ansys_deformations = np.array(ansys_deformations_)
    ansys_elem_tab = np.array(ansys_elem_tab_)

    # Get Ansys surface from Ansys deformation data
    ansys_elem_tab_surf_tri = outer_shell_tet2_tet1(ansys_elem_tab, ansys_node_coords)  # Get outer shell triangles
    ansys_elem_surf_ids = ansys_deformations[:, 0]  # IDs of elements in the analyzed surface
    ansys_elem_tab_surf = ansys_elem_tab_surf_tri[np.in1d(ansys_elem_tab_surf_tri[:, 0], ansys_elem_surf_ids)]  # Element table of the analyzed surface

    ansys_elem_tab_surf_as_args, old2new_node_ids, new2old_node_ids = elem_table_ids2args(ansys_elem_tab_surf)
    ansys_nodes_surf_coords = np.array([ansys_node_coords[np.where(ansys_node_coords == id)[0][0]] for id in
                                        list(new2old_node_ids.values())])

    # Get center point of each triangle
    cells = np.transpose((3*np.ones(len(ansys_elem_tab_surf_as_args)), *np.transpose(ansys_elem_tab_surf_as_args[:, 1:]))).astype(int)
    celltypes = 5*np.ones(len(cells))
    points = np.array(ansys_nodes_surf_coords)[:, 1:]
    cells_ravel = cells.ravel()
    surf = pyvista.UnstructuredGrid(cells_ravel, celltypes, points)
    ansys_nodes_surf_center_coords = np.ones((len(ansys_elem_tab_surf_as_args), 4))
    ansys_nodes_surf_center_coords[:, 0] = ansys_elem_tab_surf_as_args[:, 0]
    ansys_nodes_surf_center_coords[:, 1:] = surf.cell_centers().points

    # Getting ansys mesh's nodes corresponding to vic3d surface
    #   Finding parameters to position Ansys surface as Vic3D surface
    non_emty_node_mask = np.sum(vic3d_node_coords[:, 1:] != 0, axis=1).astype(bool)  # Mask of non empty nodes (with coords [0, 0, 0])
    vic3d_surf_nodes_coord = vic3d_node_coords[non_emty_node_mask]  # Removing empty nodes
    vic3d_surf_nodes_coord[:, 1:] /= 1  # Changing coordinate unit from mm to m (MUST MATCH ANSYS MESH UNIT)
    vic3d_surf_nodes_defs = vic3d_node_deformations[non_emty_node_mask]  # Deformation of non empty nodes
    rot_mat_init = R.from_euler("zyx", zyx_rot_eul_angles_init).as_matrix()
    ansys_surf_nodes_coord_matched = np.copy(ansys_nodes_surf_center_coords)
    ### DEBUG ###
    # p = pyvista.Plotter()
    # p.add_mesh(pyvista.PolyData(vic3d_surf_nodes_coord[:, 1:]), color="b")
    # p.add_mesh(pyvista.PolyData(ansys_nodes_surf_center_coords[:, 1:]), color="r")
    # p.add_mesh(pyvista.PolyData(np.mean(np.transpose(vic3d_surf_nodes_coord[:, 1:]), axis=1)), color="w")
    # p.add_mesh(pyvista.PolyData(np.mean(np.transpose(ansys_nodes_surf_center_coords[:, 1:]), axis=1)), color="k")
    # p.show()
    ### END DEBUG ###
    reg = RigidRegistration(**{'X': vic3d_surf_nodes_coord[:, 1:], 'Y': ansys_surf_nodes_coord_matched[:, 1:]}, R=None, t=None,
                            s=1, max_iterations=100, tolerance=1)
    reg.register()  # Surface matching
    s, r, t, q = reg.get_registration_parameters()  # Get scale, rotation matrix, translation vector
    # ### DEBUG ###
    # p = pyvista.Plotter()
    # p.add_mesh(pyvista.PolyData(vic3d_surf_nodes_coord[:, 1:]), color="b")
    # p.add_mesh(pyvista.PolyData(np.dot(ansys_nodes_surf_center_coords[:, 1:], r) + t), color="r")
    # p.show()
    # ### END DEBUG ###
    reg_params = [[s, r, t, q]]
    print("Registration of surfaces...")
    for orient_angles in [[np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi]]:
        for orient_angles2 in [[np.pi/2, 0, 0], [0, 0, 0]]:
            try:
                print(f"\nTrying initial orientation of surface (z->y->x) : {(np.array(orient_angles) + np.array(orient_angles2))*180/np.pi}")
                rot_mat = np.dot(reg_params[0][1], R.from_euler("zyx", np.array(orient_angles) + np.array(orient_angles2)).as_matrix())
                reg = RigidRegistration(**{'X': vic3d_surf_nodes_coord[:, 1:], 'Y': ansys_surf_nodes_coord_matched[:, 1:]},
                                        R=rot_mat, t=reg_params[0][2], s=1, max_iterations=100, tolerance=1)
                reg.register()  # Surface matching
                s, r, t, q = reg.get_registration_parameters()  # Get scale, rotation matrix, translation vector
                # ### DEBUG ###
                # p = pyvista.Plotter()
                # p.add_mesh(pyvista.PolyData(vic3d_surf_nodes_coord[:, 1:]), color="b")
                # p.add_mesh(pyvista.PolyData(np.dot(ansys_nodes_surf_center_coords[:, 1:], r) + t), color="r")
                # p.show()
                # ### END DEBUG ###
                reg_params.append([s, r, t, q])
                print(f"Done. Best orientation found (z->y->x): {R.from_matrix(r).as_euler('zyx', degrees=True)}; Q value = {q}   (smaller is better)")
            except Exception as e:
                print(e)
                print("ERROR orienting surface with angles (z->y->x) : ", (np.array(orient_angles) + np.array(orient_angles2))*180/np.pi)
    reg_params = np.array(reg_params, dtype=object)
    q_min_arg = np.argmin(reg_params[:, 3])
    s, r, t, q = reg_params[q_min_arg]
    print(f"\nBest orientation kept (z->y->x): {R.from_matrix(r).as_euler('zyx', degrees=True)}")
    print(f"Translation: {t}")

    #   Reorienting Ansys surface
    ansys_surf_nodes_coord_matched = np.copy(ansys_nodes_surf_center_coords)
    ansys_surf_nodes_coord_matched[:, 1:] = np.dot(ansys_surf_nodes_coord_matched[:, 1:], r) + t  # Repositioning of the Ansys surface

    #   Saving picture to verify surface matching
    p = pyvista.Plotter(off_screen=not show_plot)
    p.add_text(
        'Close to continue',
        position='upper_right',
        color='red',
        shadow=True,
        font_size=26
    )
    p.add_mesh(pyvista.PolyData(vic3d_surf_nodes_coord[:, 1:]), color="b")
    p.add_mesh(pyvista.PolyData(ansys_surf_nodes_coord_matched[:, 1:]), color="r")
    p.show(screenshot=save_img_path[:-4] + "_verif.png")

    #   Finding matching arg of Vic3D node with ansys node
    arg_matched_vic3d_node = []
    for coord in ansys_surf_nodes_coord_matched[:, 1:]:
        dists = np.sum((coord - vic3d_surf_nodes_coord[:, 1:]) ** 2, axis=1)  # Distance between tested Ansys node and each Vic3D nodes
        arg_matched_vic3d_node.append(np.argmin(dists))  # Node arg at min distance
    arg_matched_vic3d_node = np.array(arg_matched_vic3d_node)

    #   Getting matched Vic3D node deformation (averaged to node deformation in smoothing radius)
    vic3d_defs = []
    ansys_defs = []
    for i, arg in enumerate(arg_matched_vic3d_node):
        node_vic3d_id = vic3d_surf_nodes_coord[:, 0][arg]
        if radius_smooth > 0:
            dists = np.sqrt(np.sum((vic3d_surf_nodes_coord[:, 1:][arg] - vic3d_surf_nodes_coord[:, 1:]) ** 2, axis=1))  # Distance between tested Vic3D node and each other node
            mask_smooth = dists < radius_smooth  # Mask of nodes in radius
            vic3d_def = np.mean(vic3d_surf_nodes_defs[:, -1][mask_smooth])  # Averaging the deformation with deformation of each node in radius
        else:
            vic3d_def = np.mean(vic3d_surf_nodes_defs[:, -1][arg])

        elem_ansys_id = ansys_surf_nodes_coord_matched[i, 0]
        ansys_def = ansys_deformations[ansys_deformations[:, 0] == elem_ansys_id][0][1]  # Getting deformation of matched Ansys element
        vic3d_defs.append([node_vic3d_id, vic3d_def])
        ansys_defs.append([elem_ansys_id, ansys_def])
    vic3d_defs = np.array(vic3d_defs)
    vic3d_defs[:, 1] *= 100  # in %
    ansys_defs = np.array(ansys_defs)
    ansys_defs[:, 1] *= 100  # in %
    diff_deformation = (ansys_defs[:, 1] - vic3d_defs[:, 1]) / vic3d_defs[:, 1] * 100  # Relative diff to vic3d
    print("Mean def = ", np.mean(np.abs(diff_deformation)))

    # SAVE DATA
    wb = xlsxwriter.Workbook(save_deformation_path)
    sh = wb.add_worksheet()
    header = ["Elem id", "Ansys def (%)", "Vic3D def (%)", "(Vic3D-Ansys)/Vic3D def (%)", "Averaging radius"]
    sh.write_row(0, 0, header)
    for i in range(len(ansys_defs)):
        try:
            sh.write_row(i+1, 0, [ansys_defs[i, 0], ansys_defs[i, 1], vic3d_defs[i, 1], diff_deformation[i], radius_smooth])
        except:
            sh.write_row(i + 1, 0, [ansys_defs[i, 0], ansys_defs[i, 1], vic3d_defs[i, 1], "ERROR", radius_smooth])
    wb.close()

    # PYVISTA PLOTS
    #   Whole mesh
    tet_elem_tab = (ansys_elem_tab[:, 1:5] - 1)[~np.in1d(ansys_elem_tab[:, 0], ansys_elem_tab_surf_as_args[:, 0])]
    whole_cells = np.transpose((4 * np.ones(len(tet_elem_tab)), *np.transpose(tet_elem_tab))).astype(int)
    whole_celltypes = 10 * np.ones(len(whole_cells))
    whole_points = np.array(ansys_node_coords)[:, 1:]
    whole_cells_ravel = whole_cells.ravel()
    whole_surf = pyvista.UnstructuredGrid(whole_cells_ravel, whole_celltypes, whole_points)

    #   Plot
    plotter = pyvista.Plotter(off_screen=not show_plot)
    plotter.subplot(0, 0)
    plotter.add_mesh(whole_surf)
    plotter.add_mesh(surf, scalars=diff_deformation, cmap='jet', clim=(-100, 100))
    plotter.scalar_bar.SetTitle("%")
    normal_vec = np.mean(np.transpose(pyvista.PolyData(points, cells).cell_normals), axis=1)
    center = np.mean(np.transpose(points), axis=1)
    max_radius = np.max(np.sqrt(np.sum((center - points) ** 2, axis=1)))
    fp = center  # Focal point
    cp = center - normal_vec*1000  # Cam position
    fp_cp_dist = np.sqrt(np.sum((fp - cp) ** 2))
    viewing_angle = np.abs(np.arctan(max_radius/fp_cp_dist))*180/np.pi*2
    viewing_angle *= 1.1  # 10% margin
    plotter.camera.focal_point = fp
    plotter.camera.position = cp
    plotter.camera.view_angle = viewing_angle
    plotter.show(screenshot=save_img_path, auto_close=False)
    cp = center + normal_vec * 1000  # Opposite cam position
    if not show_plot:
        plotter.camera.position = cp
        plotter.show(screenshot=save_img_path[:-4]+"_opposite.png")



def read_nodes_def_vic3d(path_vic3d_nodes_def):
    """
    Read vic3d csv file with nodes coordinates (X, Y, Z) and Von Mises deformation.
    :param path_vic3d_nodes_def:
    :return: List of [node, coord1, coord2, coord3] from vic3d surface, List of [node, deformation] from vic3d surface.
    """

    # raw_data = []
    # with open(path_vic3d_nodes_def, 'r') as f:
    #     for line in f:
    #         raw_data.append(line.replace("\n", "").split(','))
    #     if raw_data[-1][0] == '':
    #         raw_data.pop(-1)
    with open(path_vic3d_nodes_def, 'r') as f:
        raw_data = f.read().split("\n")
    raw_data = list(map(lambda x: x.split(","), raw_data))
    while raw_data[-1][0] == '':
        raw_data.pop(-1)
    raw_data = np.array(raw_data[1:]).astype(float)
    x_arg = 0
    y_arg = 1
    z_arg = 2
    def_arg = 12
    vic3d_node_coords = np.concatenate([np.array([range(len(raw_data))]).T, raw_data[:, [x_arg, y_arg, z_arg]]], axis=1)
    vic3d_node_deformations = np.concatenate([np.array([range(len(raw_data))]).T, raw_data[:, [def_arg]]], axis=1)
    # print(vic3d_node_coords[300], vic3d_node_deformations[300])
    return vic3d_node_coords, vic3d_node_deformations


def read_nodes_coord_ansys(path_ansys_cdb):
    """
    Read ansys cdb file with nodes coordinates (X, Y, Z).
    :param path_ansys_cdb:
    :return: List of [node, coord1, coord2, coord3] from ansys mesh.
    """

    raw_data = []
    with open(path_ansys_cdb, 'r') as f:
        read_lines = False
        for line in f:
            if "NBLOCK" in line:
                read_lines = True
            if read_lines:
                if "-1," in line:
                    break
                raw_data.append(line.replace("\n", ""))
    node_id_len = int(raw_data[1].split("i")[1].split(',')[0])
    node_coord_len = int(raw_data[1].split("e")[1].split('.')[0])
    nb_id = 3
    nb_coord = 3
    raw_data = raw_data[2:]
    ansys_node_coords = list(map(lambda x: [int(x[0:node_id_len]), *[float(x[nb_id*node_id_len + i*node_coord_len:nb_id*node_id_len + (i+1)*node_coord_len]) for i in range(nb_coord)]], raw_data))
    return ansys_node_coords


def read_elem_tab_ansys(path_ansys_cdb):
    """
    Read ansys cdb file with nodes coordinates (X, Y, Z).
    :param path_ansys_cdb:
    :return: List of [element_id, node1_id, node2_id, node3_id] from ansys mesh.
    """

    raw_data = []
    with open(path_ansys_cdb, 'r') as f:
        read_lines = False
        for line in f:
            if "EBLOCK" in line:
                read_lines = True
            if read_lines:
                if "-1\n" in line:
                    break
                raw_data.append(line.replace("\n", ""))
    nb_useless_val = 10
    nb_char = int(raw_data[1].split('i')[-1].split(')')[0])
    raw_data = raw_data[2:]
    raw_data = [raw_data[2*i] + raw_data[2*i+1] for i in range(len(raw_data)//2)]
    raw_data = np.array([[raw_data[j][i*nb_char:(i+1)*nb_char] for i in range(len(raw_data[j])//nb_char)] for j in range(len(raw_data))])
    raw_data = raw_data[:, nb_useless_val:].astype(int)
    return raw_data

def read_nodes_def_ansys(path_ansys_def):
    """
    Read ansys cdb file with nodes deformation.
    :param path_ansys_cdb:
    :return: List of [node, deformation] from ansys deformation txt file.
    """

    with open(path_ansys_def, 'r') as f:
        raw_data = f.read().replace(',', '.').split('\n')[1:]
    while raw_data[-1] == '':
        raw_data.pop(-1)
    raw_data = list(map(lambda x: [int(x.split('\t')[0]), float(x.split('\t')[1])], raw_data))
    return raw_data