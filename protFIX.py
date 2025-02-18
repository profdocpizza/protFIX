from utils import (
    align_input_sequence_to_residue_df,
    check_eligibility,
    generate_residue_df,
    load_tasks,
    get_root_dir,
    pack_side_chains_on_diffusion_output,
    run_structure_inpainting,
)
import ampal
import os

tasks = load_tasks(os.path.join(get_root_dir(), "tasks.yaml"))


def run_task(task):

    assembly = ampal.load_pdb(task["pdb_path"])
    check_eligibility(assembly)
    # TODO check if qualifies and is supported

    chain_id = assembly[0].id
    monomers = list(assembly[0].get_monomers())

    residue_df = generate_residue_df(monomers)
    residue_df = align_input_sequence_to_residue_df(residue_df, task["sequence"])
    selected_sequence = residue_df["complete_mol_letter"].to_list()

    diffusion_output_file_path = run_structure_inpainting(
        task, residue_df, monomers, chain_id
    )

    packed_file_path = pack_side_chains_on_diffusion_output(
        diffusion_output_file_path, selected_sequence
    )

    print(f"Task {task['name']} done. PDB saved at {packed_file_path}")


def main():
    for task in tasks:
        run_task(task)


if __name__ == "__main__":
    main()
