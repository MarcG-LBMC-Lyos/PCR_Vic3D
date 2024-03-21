import PCR_Vic3D_elem
import os
import zipfile

vic3d_data_path = [r".\Test_res\def_init_vic3d.csv", r".\Test_res\def_final_vic3d.csv"]
ansys_cdb_path = r".\Test_res\mesh.cdb"  # UNZIP THE MESH FILE IN "TEST_RES" FOLDER BEFORE RUNNING THE SCRIPT
ansys_deformations_path = [r".\Test_res\MaxPrincipalStrain_meshSurface.txt"]
if not os.path.exists(ansys_cdb_path):
    with zipfile.ZipFile(r".\Test_res\mesh.zip", 'r') as zip_ref:
        zip_ref.extractall(r".\Test_res")

dirs = []
dirs.append({"vic3d": vic3d_data_path,
             "ansys_def": ansys_deformations_path,
             "cdb": ansys_cdb_path})


for paths in dirs:
    Vic3D_nodes_coord_init, Vic3D_nodes_deformations_init = PCR_Vic3D_elem.read_nodes_def_vic3d(paths["vic3d"][0])
    Vic3D_nodes_coord, Vic3D_nodes_deformations = PCR_Vic3D_elem.read_nodes_def_vic3d(paths["vic3d"][1])
    Vic3D_nodes_deformations[:, 1] -= Vic3D_nodes_deformations_init[:, 1]

    Ansys_nodes_coord = PCR_Vic3D_elem.read_nodes_coord_ansys(paths["cdb"])
    Ansys_elem_tab = PCR_Vic3D_elem.read_elem_tab_ansys(paths["cdb"])

    for ansys_deformations_path in paths["ansys_def"]:
        try:
            ansys_deformations = PCR_Vic3D_elem.read_nodes_def_ansys(ansys_deformations_path)

            save_img_path = os.path.splitext(ansys_deformations_path)[0] + ".png"
            save_deformation_path = os.path.splitext(ansys_deformations_path)[0] + "_compareVic3D.xlsx"

            # zyx_rot_eul_angles_init = (np.pi, -np.pi/2, 0)
            zyx_rot_eul_angles_init = (0, 0, 0)
            radius_smooth = 0.0005
            PCR_Vic3D_elem.pcr_vic3d(Vic3D_nodes_coord, Ansys_nodes_coord, Vic3D_nodes_deformations, ansys_deformations,
                                     Ansys_elem_tab,
                                     zyx_rot_eul_angles_init=zyx_rot_eul_angles_init,
                                     radius_smooth=radius_smooth,
                                     show_plot=True,
                                     save_img_path=save_img_path,
                                     save_deformation_path=save_deformation_path)
        except Exception as e:
            print(e)
            print("ERROR with file : ", ansys_deformations_path)

